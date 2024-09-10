import express, { Request, Response } from 'express';
import multer from 'multer';
// TO DO: use @tensorflow/tfjs-node instead of canvas: https://www.tensorflow.org/js/guide/nodejs
import * as tfjs from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';
import cors from 'cors';
import { createCanvas, loadImage } from 'canvas'; // Import canvas functions

console.log(tfjs.version.tfjs);
// import dotenv from 'dotenv';
// dotenv.config();

const app = express();
const port = 5000;
app.listen(port, () => console.log(`Server is running on http://localhost:${port}`));

const upload = multer({ dest: 'uploads/' }); // Ensure 'uploads' folder exists

app.use('/models', express.static('models'));

let model: tfjs.LayersModel | null = null;
async function loadModel() {
  try {
    model = await tfjs.loadLayersModel('http://localhost:5000/../models/model.json');
    console.log('Model is running on http://localhost:5000/../models/model.json');
  } catch (error) {
    console.error('Failed to load the model:', error);
  }
}

// Enable CORS for all routes
app.use(cors());

// Ensure the model is loaded before the app starts listening
loadModel();

// Handle file upload and prediction
app.post('/predict', upload.single('image'), async (req: Request, res: Response) => {
  const imagePath = req.file?.path;

  if (!imagePath) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  if (!model) {
    return res.status(500).json({ error: 'Model not loaded' });
  }

  try {
    const result = await predict(imagePath);
    res.json({ prediction: result });
  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).json({ error: 'Prediction failed' });
  } finally {
    // Clean up the uploaded file
    fs.unlinkSync(imagePath);
  }
});

const predict = async (imagePath: string) => {
  try {
    // Load and preprocess the image
    const image = await loadImage(imagePath);
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 224, 224); // Resize to 224x224

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixelData = new Uint8Array(imageData.data.buffer);

    // Convert RGBA to Grayscale
    const grayPixelData = new Uint8Array(224 * 224); // Initialize an array for grayscale data
    for (let i = 0; i < pixelData.length; i += 4) {
      const r = pixelData[i];
      const g = pixelData[i + 1];
      const b = pixelData[i + 2];
      // Use the luminosity method to convert to grayscale
      grayPixelData[i / 4] = (r * 0.299 + g * 0.587 + b * 0.114); // Luminosity method
    }

    const tfImage = tfjs.tensor3d(grayPixelData, [224, 224, 1]); // Create a tensor with shape [224, 224, 1]

    const normalizedImage = tfImage.div(tfjs.scalar(255.0)).expandDims(0); // Normalize to [0, 1]

    // Make predictions
    const predictions = model!.predict(normalizedImage) as tfjs.Tensor; // Ensure model is non-null
    const scores = await predictions.array() as number[][];
    const classIndex = scores[0][0] > scores[0][1] ? 0 : 1; // 0: Negative, 1: Positive
    return classIndex === 1 ? 'Tumor Detected' : 'Tumor Not Detected';
  } catch (error) {
    console.error('Error during image processing or prediction:', error);
    throw new Error('Prediction failed'); // Re-throw the error for handling in the route
  }
};

