/* TO DO: find a way to use @tensorflow/tfjs-node instead of canvas: 
https://www.tensorflow.org/js/guide/nodejs */
import * as tfjs from '@tensorflow/tfjs';
import { loadImage } from 'canvas';
import { preprocessImage } from './image.service.ts';

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

export const predict = async (imagePath: string): Promise<string> => {
  if (!model) {
    throw new Error('Model not loaded');
  }

  const image = await loadImage(imagePath);
  const processedImage = preprocessImage(image);
  const predictions = model.predict(processedImage) as tfjs.Tensor;
  const results = await predictions.array() as number[][];
  return results[0][0] > results[0][1] ? 'No tumour detected' : 'Tumour detected'; // 0 = -ve, 1 = +ve
  // const results = await prediction.dataSync()[0] > 0.5 ? 'Tumor detected' : 'No tumor detected';
  predictions.dispose();
};
