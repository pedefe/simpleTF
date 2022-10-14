## simpleTF, a simple sample for classification with tensorflow and NodeJS

### Introduction
I tried to use Tensorflow.js to analyse pictures and to recognize vehicles types.
So I wrote a simple code in html/js to do that, but I could'nt succed to load/save the model, and load/save the classifier.
After one day, I decide to switch on NodeJS to get all freedom to access to disk.
I'm under Windows.
So here is a unique code to :
- load nodejs modules;
- create new model or load existing one;
- create new knn-classifier or load existing one;
- test image if classifier exist;
- learn images with their known class;
- test image after learning;

Images are stored in a ./datas sub-folder, and their names include the class.

### Configuration
	1.   Install NodeJS
	2.   Create folder
	3.   In folder, create project
	`> npm create`
	4.   Install modules with npm:
	`> npm install fs`
	`> npm install @tensorflow/tfjs`
	`> npm install @tensorflow/tfjs-node`
	`> npm install @tensorflow-models/mobilnet`
	`> npm install @tensorflow-models/knn-classifier`

	5.    You need to create a subfolder call ./datas, containing png pictures, like :
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

	6.    Then, you can run the code:
	`node simpleTF`
	
### What the program does
Why lot of operation are asynchronous, you need to know how async/await is working...
My sample works on vehicle pictures taken from front view. So my 'modelName' is named 'vehicleFront'.

The several steps are the following. These code works well and can be a good base for use of Tensorflow in NodeJS.

	async function process() {
		// name of set of images (vehicles pictures captured from front side)
		let modelName = "vehicleFront";

		// initialize network
		// try to load previous saved model
		// try to load previous classifier
		await initializeNet( modelName);

		// load all pictures (there are around ten, we are in a sample...)
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
