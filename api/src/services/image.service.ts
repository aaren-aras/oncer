import { createCanvas, Canvas, Image } from 'canvas';
import * as tfjs from '@tensorflow/tfjs';

export const preprocessImage = (image: Canvas | Image) => {
  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 224, 224); // resize to 224x224

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixelData = new Uint8Array(imageData.data.buffer);

  // Convert RGBA to greyscale
  const greyPixelData = new Uint8Array(224 * 224);
  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i], g = pixelData[i + 1], b = pixelData[i + 2];
    greyPixelData[i / 4] = r * 0.299 + g * 0.587 + b * 0.114; // luminosity method
  }

  // Create a tensor with shape [224, 224, 1] and normalize to [0, 1]
  return tfjs.tensor3d(greyPixelData, [224, 224, 1]).div(tfjs.scalar(255.0)).expandDims(0);
}