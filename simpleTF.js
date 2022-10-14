/*	simpleTF.js
	¤ NodeJS program
	¤ you need to install NodeJs
	¤ you need to install with npm:
		> npm install @tensorflow/tfjs
		> npm install @tensorflow/tfjs-node
		> npm install @tensorflow-models/mobilnet
		> npm install @tensorflow-models/knn-classifier

	¤ you need a subfolder call ./datas, containing png pictures, like :
		bike.1.png
		bike.2.png
		...
		bike.5.png
		car.1.png
		car.2.png
		...
		car.6.png
		motorbike.1.png
		...
		motorbike.3.png
		truck.1.png
		...
		truck.3.png

	*/
var _appName = "simpleTF";

/**/
function log( context, text) {
	console.log( `${new Date().toISOString()} > ${context} > ${text}`);
}

log( _appName,  `starts...`);

// npm install @tensorflow/tfjs
const _tf = require('@tensorflow/tfjs');
// npm install @tensorflow/tfjs-node
const _tfnode = require('@tensorflow/tfjs-node');
log( _appName, `tf loaded.`);

// npm install @tensorflow-models/mobilnet
const _mobilenet = require('@tensorflow-models/mobilenet');
log( _appName, `mobilenet loaded.`);

// npm install @tensorflow-models/knn-classifier
const _knnClassifier = require('@tensorflow-models/knn-classifier');
log( _appName, `knn-classifier loaded.`);

const fs = require('fs');

var _classLabels = [ "car", "bike", "truck", "motorbike", "other"];

var _myNet;
var _myClassifier;
var _learnArray = [];
var _picturesArray = [];
var _classifierFound = false;
var _ll = false;	// low-level log

/*
	*/
async function initializeNet( modelName) {
	const fn = "initializeNet";
	do {
		if (_ll) log( fn, "create virgin model...");
		try {
			_myNet	= await _mobilenet.load();
			log( fn, "virgin model created.");
		} catch( err) {
			log( fn, `create virgin model fails. err: ${err}`);
			break;
		}

		if (_ll) log( fn, "create classifier...");
		try {
			_myClassifier = _knnClassifier.create();
			log( fn, "classifier created.");
		} catch( err) {
			log( fn, `create classifier fails. err: ${err}`);
			break;
		}

		let folder = `./my-model-${modelName}`;
		if ( ! fs.existsSync( `${folder}/model.json`)
			|| ! fs.existsSync( `${folder}/weights.bin`)) {
			log( fn, `no previous model ${modelName} found...`);
			}
		else {

			try {
				if (_ll) log( fn, `load previous model...`);
				await _myNet.model.load(`file://${folder}`);
				log( fn, `previous model '${modelName}' loaded`);
			} catch( err) {
				log( fn, `load previous model  '${modelName}' fails. err: ${err}`);
				break;
			}

			let savedDataSet = getSavedClassifier( modelName);
			if (savedDataSet) {
				try {
					if (_ll) log( fn, `set previous classifier...`);
					_myClassifier.setClassifierDataset( savedDataSet);
					log( fn, `previous classifier set.`);
					_classifierFound = true;
				} catch( err) {
					log( fn, `load previous classifier fails. err: ${err}`);
				}
			} else {
				log( fn, `no previous classifier found.`);
			}

		}
	} while( false);
}

/* saved knnClassifier on file
	*/
function saveClassifier( modelName) {
	const fn = "saveClassifier";

  let dataset = _myClassifier.getClassifierDataset();
  let datasetObj = {}
  Object.keys(dataset).forEach((key) => {
    let data = dataset[key].dataSync();
    datasetObj[key] = Array.from(data);
  });
  let jsonStr = JSON.stringify(datasetObj)
	let folder = `./my-model-${modelName}`;
	let classifierFileName = `${folder}/classifier.json`
  fs.writeFileSync( classifierFileName, jsonStr);
	if (_ll) log( fn, `classifier ${modelName} saved`);
}

/* load knnClassifier from file
	*/
function getSavedClassifier( modelName) {
	const fn = "getSavedClassifier";
	let tensorObj = undefined;
	let folder = `./my-model-${modelName}`;
	let classifierFileName = `${folder}/classifier.json`
	if (fs.existsSync( classifierFileName)) {
		let dataset = fs.readFileSync( classifierFileName)
		tensorObj = JSON.parse(dataset)
	  Object.keys(tensorObj).forEach((key) => {
			tensorObj[key] = _tf.tensor( tensorObj[key], [tensorObj[key].length / 1024, 1024]);
	  });
		if (_ll) log( fn, `classifier ${modelName} loaded`);
	}
	return tensorObj;
}

/*
	*/
function declareNewPicture( classId, fileName, image) {
	if ( ! _picturesArray[ fileName]) {
		_picturesArray[ fileName] = {
			fileName: fileName,
			classId: classId,
			image: image,
			results: []
		}
	}
}


/* declare image as classId
	*/
async function addExample(classId, img, fileName) {
	const fn = "addExample";
	
	// Get the intermediate activation of MobileNet 'conv_preds' and pass that
	// to the KNN classifier.
	const activation = _myNet.infer( img, true);

	// Pass the intermediate activation to the classifier.
	_myClassifier.addExample( activation, classId);
			
	let pictureInfo = _picturesArray[ fileName];
	let trainResult = {
		stamp: new Date().toISOString(),
		step: "training", 
		classId: classId,
		classLabel: _classLabels[ classId]
	};
	pictureInfo.results.push( trainResult);

	if (_ll) log( fn, `modelName '${fileName}' recorded as '${_classLabels[ classId]}'`);
}

/* load all known pictures
	*/
async function loadAllPictures( path) {
	const fn = "loadAllPictures";
	
	return new Promise( async (resolve) => {
		for( let classId=0; classId < _classLabels.length; classId++) {
			let classLabel = _classLabels[ classId];
			let index= 1;
			let run = true;
			while( true) {
				try {
					let result = await loadPicture( path, classId, index);
					if (result.success == false) {
						// pas d'autres fichiers de ce 'type'
						break;
					}
				}
				catch(err) {
					log( fn, `local exception. err: ${err}`);
					break;
				}
				index++;
			}
		}
		resolve();
	});
}

/* lear all images with there known classId
	*/
async function learnImages() {
	const fn = "learnImages";
	
	return new Promise( async (resolve) => {

		for( let i=0; i < Object.keys( _picturesArray).length; i++) {
			let fileName = Object.keys( _picturesArray)[ i];
			let pictureInfo = _picturesArray[ fileName];

			// apprend l'image en tant que classId
			await addExample( pictureInfo.classId, pictureInfo.image, pictureInfo.fileName);
		}
		resolve();
	});
}

/* Test all images of datas folder
	*/
async function testImages() {
	let index;
	
	return new Promise( async (resolve) => {
		for( let i=0; i < Object.keys( _picturesArray).length; i++) {
			let fileName = Object.keys( _picturesArray)[ i];
			let pictureInfo = _picturesArray[ fileName];

			await analyseLearnedImage( pictureInfo.image, pictureInfo);
		}
		resolve();
	});
}

/* how to load a picture from file to tensorflow input format
	*/
async function loadPicture( path, classId, index) {
	const fn = "loadPicture";

	let classLabel = _classLabels[ classId];
	let name = `${classLabel}.${index}`;
	let fileName = `${path}/${name}.png`;	
	return new Promise( (resolve) => {
		if (fs.existsSync( fileName)) {
			const data = fs.readFileSync( fileName);
			let image;
			try {
				// image = _tf.node.decodeImage( Buffer.concat(data));
				image = _tfnode.node.decodeImage( data);
				if (_ll) log( fn, `${fileName} loaded.`);
			} catch( err) {
				log( fn, `${fileName} > err: ${err}`);
			}
			if (image) {
				
				declareNewPicture( classId, fileName, image);

				resolve( { success: true, fileName: fileName });
			} else {
				resolve( { success: false, fileName: fileName });
			}
		} else {
			resolve( { success: false, fileName: fileName });
		}
	});
}
		
/* analyse image, depending if classifier exist or not
	*/
async function analyseLearnedImage( img, pictureInfo) {
	const fn = "analyseLearnedImage";

	return new Promise( async function request(resolve) {
		if (_myClassifier.getNumClasses() > 0) {
			try {
				if (_ll) log( fn, `request class...`);
				// Get the activation from mobilenet
				const activation = _myNet.infer( img, true);
				
				// Get the most likely class and confidence from the classifier module.
				const K = 3;
				const result = await _myClassifier.predictClass( activation, K);

				let classId = result.label;
				let classScore = 0;
				Object.keys( result.confidences).forEach( kId => {
					if (result.confidences[ kId] > classScore) {
						classScore = result.confidences[ kId];
						classId = parseInt( kId);
					}
				});

				let analyseResult = {
					stamp: new Date().toISOString(),
					step: "analyse", 
					classId: parseInt( result.label),
					classLabel: _classLabels[ parseInt(result.label)],
					confidence: result.confidences[ result.label]
				};
				pictureInfo.results.push( analyseResult);
										
				if (_ll) log( fn, `classification of '${pictureInfo.fileName}': ${JSON.stringify( result)}`);
			} 
			catch( err) {
				log( fn, `${pictureInfo.fileName} > err: ${err}`);
			}
		} else {
			if (_ll) log( fn, `${pictureInfo.fileName} > no classes`);
			try {
				if (_ll) log( fn, `${pictureInfo.fileName} > request class...`);
				let predictions = await _myNet.classify( img);

				predictions.forEach( result => {
					if (result.probability) result.probability = result.probability.decimale( 2);
				});
				// first prediction
				if (predictions.length > 1) {
					predictions = predictions.splice( 0, 1);
				}
				if (predictions.length > 0) {
					
					let analyseResult = {
						stamp: new Date().toISOString(),
						step: "analyse", 
						classId: undefined,
						classLabel: predictions[0].className,
						confidence: predictions[0].probability
					};
					pictureInfo.results.push( analyseResult);

					if (_ll) log( fn, `${pictureInfo.fileName} > ${JSON.stringify( predictions[0])}`);
				} else {
					log( fn, `${pictureInfo.fileName} > no prediction`);
				}
			} 
			catch( err) {
				log( fn, `${pictureInfo.fileName} > err: ${err}`);
			}
		}
		resolve();
	});
}

/* cut decimale (when confidence float are too long)
	*/
if (Number.prototype.decimale === undefined) {
	Number.prototype.decimale = function( nb) {
		return Number.parseFloat( Number.parseFloat( this).toFixed( nb));
	}
}

/* Save and free tf ressources
	*/
async function saveAndFreeAll( modelName) {
	const fn = "freeAll";

	if (_ll) log( fn, `saving model ${modelName}...`);
	
	try {
		await _myNet.model.save(`file://./my-model-${modelName}`);
		log( fn, `model '${modelName}' saved.`);
	} catch( err) {
		log( fn, `save model '${modelName}' fails. err: ${err}`);
	}

	saveClassifier( modelName );

	// save classifier...
	if (_myClassifier
		&& _myClassifier.dispose) _myClassifier.dispose();

	if (_myNet
			&& _myNet.dispose) _myNet.dispose();

	// Dispose the tensor to release the memory.
	Object.keys( _picturesArray).forEach( fileName => {
		let pictureInfo = _picturesArray[ fileName];
		if (pictureInfo.image) {
			pictureInfo.image.dispose();
			pictureInfo.image = undefined;
		}
	});
}

/*
	*/
function displayResults( modelName, onFile, onConsole) {
	const fn = "displayResults";

	let sContent = "";
	if (onConsole) log( fn, "Report...");
	
	Object.keys( _picturesArray).forEach( fileName => {
		let pictureInfo = _picturesArray[ fileName];
		if (pictureInfo.image) {
			pictureInfo.image.dispose();
			pictureInfo.image = undefined;
		}
		if (onConsole) console.log( JSON.stringify( pictureInfo, null, '\t'));
		if (onFile) sContent += JSON.stringify( pictureInfo, null, '\t') + "\n";
	});
	if (onConsole) log( fn, "Report.End.");
	if (onFile) {
		let fileName = `./${modelName}.report.txt`;
		fs.writeFileSync( fileName, sContent);
		log( fn, `report saved on file '${fileName}'`);
	}
}

/* Whole passes...
	*/
async function process() {
	
	// name of set of images (vehicles pictures captured from front side)
	let modelName = "vehicleFront";

	// initialize network
	// try to load previous saved model
	// try to load previous classifier
	await initializeNet( modelName);

	await loadAllPictures( "./datas");

	// If classifier found, test images
	if (_classifierFound) {
		await testImages();
	}

	// learn image with known 'classId'
	await learnImages();

	// re-test all images
	await testImages();

	// reporting
	displayResults( modelName, true, false);

	// save and free model and classifier
	await saveAndFreeAll( modelName);

	log( _appName, "application ends.");
}


process();
