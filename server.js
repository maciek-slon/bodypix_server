// REST server
var express = require('express');
var multer = require('multer');

// body parts segmentation
var bodyPix = require('body-pix-node');
var tfjs = require('@tensorflow/tfjs-node');

// files handling
var pngjs = require('pngjs');
var fs = require('fs');
var path = require('path');

// create web application and uploader 
var app = express();

const upload = multer({
  dest: path.join(__dirname, "./uploads")
  // you might also want to set some limits: https://github.com/expressjs/multer#limits
});


const modelUrl = 'file://' + path.join(__dirname, './models/bodypix_resnet50_float_model-stride16/model.json');

var net;

// load image to TF tensor
async function loadImage(path) {
  const file = await fs.promises.readFile(path);

  const image = await tfjs.node.decodeImage(file, 3);

  return image;
}

// save segmentation result to B/W image (mask)
async function saveSegmentationToFile(segmentation, path) {
  const pngImage = new pngjs.PNG({
    colorType: 0, 
    width: segmentation.width, 
    height: segmentation.height, 
    inputColorType: 0,
    data: segmentation.data
  });

  pngImage.data = Buffer.from(segmentation.data.map(pixel => pixel * 255));

  return new Promise((resolve) => {
    pngImage.pack().pipe(fs.createWriteStream(path)).on('close', () => resolve());
  });
}


async function estimate(fpath) {
  if (net == null) {
    net = await bodyPix.load({
      architecture: 'ResNet50',
      quantBytes: 1,
      outputStride: 16,
      modelUrl: modelUrl,
    });
  }

  const image = await loadImage(fpath);

  var fname = path.parse(fpath).name;

  const personSegmentation = await net.segmentPerson(image, {
    internalResolution: 'full',
    segmentationThreshold: 0.5
  });

  personSegmentation.data.map(value => value * 255)

  respath = './results/' + fname + '.png';
  await saveSegmentationToFile(personSegmentation, respath);

  return path.join(__dirname, respath);
}


// main page - uploader form
app.get('/', express.static(path.join(__dirname, './public')))

// segmentation
app.post('/upload', upload.single("file"), async function (req, res) {
  if (req.file != null) {
    result = await estimate(req.file.path);
    res.sendFile(result);
  } else {
    res.status(400).contentType('text/plain').end('No image provided');
  }
})

// run application server
var server = app.listen(8081, function () {
   var host = server.address().address
   var port = server.address().port
   console.log("Example app listening at http://%s:%s", host, port)
})
