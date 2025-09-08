/* TO DO: find a way to use @tensorflow/tfjs-node instead of tfjs: 
https://www.tensorflow.org/js/guide/nodejs */
import * as tfjs from '@tensorflow/tfjs';
import { preprocessImage, preprocessModalities, overlaySegmentation } from './image.service.ts';

// console.log(tfjs.version.tfjs);
let model: tfjs.LayersModel | null = null;

export const loadModel = async () => {
  try {
    model = await tfjs.loadLayersModel('http://localhost:5000/../models/model.json');
    console.log('Model loaded successfully! Running on: http://localhost:5000/../models/model.json');
  } catch (e) {
    console.error('Failed to load model:', e);
  }
}

type ModalityPaths = { t1: string; t1ce: string; t2: string; flair: string };

export const predict = async (modalities: ModalityPaths): Promise<{ prediction: string, overlay: string }> => {
  if (!model) throw new Error('Model not loaded');

  const { tensor } = await preprocessModalities(modalities);
  const output = model.predict(tensor) as tfjs.Tensor | tfjs.Tensor[];

  const [segTensor, classTensor] = output as [tfjs.Tensor4D, tfjs.Tensor2D];
  const classProb = await classTensor.data();
  const tumourPresent = classProb[1] > classProb[0];

  const mask = segTensor.squeeze().argMax(-1) as tfjs.Tensor2D;
  const overlay = await overlaySegmentation(mask);

  // Cleanup
  mask.dispose(); segTensor.dispose(); classTensor.dispose(); tensor.dispose();

  return {
    prediction: tumourPresent ? 'Tumour(s) detected' : 'No tumours detected',
    overlay
  };

// export const predict = async (filePath: string, mimetype: string): Promise<{prediction: string, overlay: string}> => {
//   if (!model) throw new Error('Model not loaded');

//   const { tensor } = await preprocessImage(filePath, mimetype);
//   const output = model.predict(tensor) as tfjs.Tensor | tfjs.Tensor[];
//   // console.log("Predicted mask shape:", output.shape);
//   // console.log("Max value in mask:", tf.max(output).dataSync()[0]);

//   // if(!Array.isArray(output)) throw new Error('Expected multiple outputs');

//   const [segTensor, classTensor] = output as [tfjs.Tensor4D, tfjs.Tensor2D];

//   const classProb = await classTensor.data();
//   console.log('Class probabilities:', classProb);
//   const tumourPresent = classProb[1] > classProb[0];

//   const mask = segTensor.squeeze().argMax(-1) as tfjs.Tensor2D;
//   const overlay = await overlaySegmentation(mask);

//   mask.dispose();
//   segTensor.dispose();
//   classTensor.dispose();
//   tensor.dispose(); 

//   return {
//     prediction: tumourPresent ? 'Tumour(s) detected' : 'No tumours detected',
//     overlay
//   }



  /* const image = await loadImage(imagePath);
  const processedImage = preprocessImage(image);
  const predictions = model.predict(processedImage) as tfjs.Tensor;
  const results = await predictions.array() as number[][];
  return results[0][0] > results[0][1] ? 'No tumours detected' : 'Tumour(s) detected'; // 0 = -ve, 1 = +ve
  // const results = await prediction.dataSync()[0] > 0.5 ? 'Tumor detected' : 'No tumor detected';
  predictions.dispose(); */

  /*
  const image = await loadImage(imagePath);
  const canvas = createCanvas(240, 240);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 240, 240);

  // Greyscale conversion
  const imageData = ctx.getImageData(0, 0, 240, 240);
  const grey = new Float32Array(240 * 240);
  for (let i = 0; i < grey.length; i++) {
    const [r, g, b] = [imageData.data[i * 4], imageData.data[i * 4 + 1], imageData.data[i * 4 + 2]];
    grey[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
  }
  const tensor = tfjs.tensor4d(grey, [1, 240, 240, 1]);

  const output = model.predict(tensor) as tfjs.Tensor;
  const mask = output.squeeze().argMax(-1); // (240, 240) int labels
  const maskArray = await mask.array();

  // 🎨 Generate overlay
  const overlayCanvas = createCanvas(240, 240);
  const octx = overlayCanvas.getContext('2d');
  const overlayImage = octx.createImageData(240, 240);
  for (let y = 0; y < 240; y++) {
    for (let x = 0; x < 240; x++) {
      const label = maskArray[y][x];
      const i = (y * 240 + x) * 4;
      let [r, g, b] = [0, 0, 0];
      if (label === 1) [r, g, b] = [255, 0, 0];       // NCR (red)
      if (label === 2) [r, g, b] = [0, 255, 0];       // ED (green)
      if (label === 3) [r, g, b] = [0, 0, 255];       // ET (blue)
      overlayImage.data[i] = r;
      overlayImage.data[i + 1] = g;
      overlayImage.data[i + 2] = b;
      overlayImage.data[i + 3] = label ? 120 : 0;     // transparency
    }
  }
  octx.putImageData(overlayImage, 0, 0);

  const base64 = overlayCanvas.toDataURL();
  const tumorPresent = mask.max().arraySync() > 0;
  return { prediction: tumorPresent ? 'Tumour(s) detected' : 'No tumour detected', overlay: base64 };
  */

};
