import { createCanvas, loadImage, Canvas, Image } from 'canvas';
import * as tfjs from '@tensorflow/tfjs';
import * as nifti from 'nifti-reader-js';
import fs from 'fs';
import npy from 'npyjs'; // install with: npm install npyjs

// export const preprocessImage = (image: Canvas | Image) => {
//   const canvas = createCanvas(224, 224);
//   const ctx = canvas.getContext('2d');
//   ctx.drawImage(image, 0, 0, 224, 224); // resize to 224x224

//   const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
//   const pixelData = new Uint8Array(imageData.data.buffer);

//   // Convert RGBA to greyscale
//   const greyPixelData = new Uint8Array(224 * 224);
//   for (let i = 0; i < pixelData.length; i += 4) {
//     const r = pixelData[i], g = pixelData[i + 1], b = pixelData[i + 2];
//     greyPixelData[i / 4] = r * 0.299 + g * 0.587 + b * 0.114; // luminosity method
//   }

//   // Create a tensor with shape [224, 224, 1] and normalize to [0, 1]
//   return tfjs.tensor3d(greyPixelData, [224, 224, 1]).div(tfjs.scalar(255.0)).expandDims(0);
// }

// export const preprocessImage = async (filePath: string): Promise<{ tensor: tfjs.Tensor4D, metadata: any }> => {
//   const ext = filePath.split('.').pop()?.toLowerCase();

//   if (ext === 'nii' || ext === 'gz') {
//     const buffer = fs.readFileSync(filePath);
//     const isCompressed = nifti.isCompressed(buffer);
//     const data = nifti.readHeader(isCompressed ? nifti.decompress(buffer) : buffer);
//     const img = nifti.readImage(data, isCompressed ? nifti.decompress(buffer) : buffer);
//     const raw = new Float32Array(img);
//     const dims = data.dims;

//     const midSlice = Math.floor(dims[3] / 2); // Axial mid slice
//     const imgTensor = tfjs.tensor4d(raw, [dims[1], dims[2], dims[3], 1]);
//     const slice = imgTensor.slice([0, 0, midSlice, 0], [dims[1], dims[2], 1, 1]);
//     const resized = tfjs.image.resizeBilinear(slice, [240, 240]);
//     const normalized = resized.div(255.0).expandDims(0) as tfjs.Tensor4D;
//     return { tensor: normalized, metadata: { dims: dims.slice(1, 4) } };
//   }

//   const img = await loadImage(filePath);
//   const canvas = createCanvas(240, 240);
//   const ctx = canvas.getContext('2d');
//   ctx.drawImage(img, 0, 0, 240, 240);
//   const imageData = ctx.getImageData(0, 0, 240, 240);

//   const grey = new Uint8Array(240 * 240);
//   for (let i = 0; i < imageData.data.length; i += 4) {
//     const [r, g, b] = [imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]];
//     grey[i / 4] = 0.299 * r + 0.587 * g + 0.114 * b;
//   }

//   const tensor = tfjs.tensor3d(grey, [240, 240, 1]).div(255).expandDims(0) as tfjs.Tensor4D;
//   return { tensor, metadata: { dims: [240, 240, 1] } };
// };



export const preprocessNpy = async (filePath: string): Promise<{ tensor: tfjs.Tensor4D, metadata: any }> => {
  const npyjs = new npy();

  const buffer = fs.readFileSync(filePath);
  const npArray = await npyjs.parse(buffer.buffer); // { data: TypedArray, shape }

  const { data, shape } = npArray; // shape: [240, 240, 4]

  // Type guard
  if (!(data instanceof Float32Array || data instanceof Uint8Array || data instanceof Int32Array)) {
    throw new Error(`Unsupported data type: ${Object.prototype.toString.call(data)}`);
  }

  const plainArray = Array.from(data); // Safe conversion
  const fullShape: [number, number, number, number] = [1, ...shape] as [number, number, number, number];

  const tensor = tfjs.tensor4d(plainArray, fullShape);
  return {
    tensor,
    metadata: { dims: shape }
  };
};


export const preprocessModalities = async (paths: { t1: string, t1ce: string, t2: string, flair: string }) => {
  const loadGreyTensor = async (path: string): Promise<tfjs.Tensor3D> => {
    const img = await loadImage(path);
    const canvas = createCanvas(240, 240);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 240, 240);
    const imageData = ctx.getImageData(0, 0, 240, 240);

    const grey = new Uint8Array(240 * 240);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const [r, g, b] = [imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]];
      grey[i / 4] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // return tfjs.tensor3d(grey, [240, 240, 1]).div(255);
    return tfjs.tensor3d(grey, [240, 240, 1]).div(255) as tfjs.Tensor3D;

  };

  const [t1, t1ce, t2, flair] = await Promise.all([
    loadGreyTensor(paths.t1),
    loadGreyTensor(paths.t1ce),
    loadGreyTensor(paths.t2),
    loadGreyTensor(paths.flair)
  ]);

  const stacked = tfjs.stack([t1, t1ce, t2, flair], -1).reshape([1, 240, 240, 4]) as tfjs.Tensor4D;

  return {
    tensor: stacked,
    metadata: { dims: [240, 240, 4] }
  };
};


export const preprocessImage = async (filePath: string, mimetype: string): Promise<{ tensor: tfjs.Tensor4D, metadata: any }> => {
  const ext = filePath.split('.').pop()?.toLowerCase();

  if (ext === 'npy') {
    console.log('entering npy path')
    return await preprocessNpy(filePath);
  }

  if (mimetype.includes('nii') || filePath.endsWith('.nii') || filePath.endsWith('.nii.gz')) {
  // if (ext === 'nii' || ext === 'gz') {
    const buffer = fs.readFileSync(filePath);
    const isCompressed = nifti.isCompressed(buffer);
    const data = nifti.readHeader(isCompressed ? nifti.decompress(buffer) : buffer);
    const img = nifti.readImage(data, isCompressed ? nifti.decompress(buffer) : buffer);
    const raw = new Float32Array(img);
    const dims = data.dims;

    const midSlice = Math.floor(dims[3] / 2); // Axial mid slice
    const imgTensor = tfjs.tensor4d(raw, [dims[1], dims[2], dims[3], 1]);
    const slice = imgTensor.slice([0, 0, midSlice, 0], [dims[1], dims[2], 1, 1]);
    const resized = tfjs.image.resizeBilinear(slice, [240, 240]);
    const normalized = resized.div(255.0).expandDims(0) as tfjs.Tensor4D;
    return { tensor: normalized, metadata: { dims: dims.slice(1, 4) } };
  }

  const img = await loadImage(filePath);
  const canvas = createCanvas(240, 240);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, 240, 240);
  const imageData = ctx.getImageData(0, 0, 240, 240);

  const grey = new Uint8Array(240 * 240);
  for (let i = 0; i < imageData.data.length; i += 4) {
    const [r, g, b] = [imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]];
    grey[i / 4] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // const tensor = tfjs.tensor3d(grey, [240, 240, 1]).div(255).expandDims(0) as tfjs.Tensor4D;
  // return { tensor, metadata: { dims: [240, 240, 1] } };

  // const tensor1 = tfjs.tensor3d(grey, [240, 240, 1]).div(255);
  // const tensor4 = tfjs.tile(tensor1.expandDims(0), [1, 1, 1, 4]) as tfjs.Tensor4D;

  // return { tensor: tensor4, metadata: { dims: [240, 240, 4] } };

  const tensor1 = tfjs.tensor3d(grey, [240, 240, 1]).div(255); // [240, 240, 1]
  const blank = tfjs.zerosLike(tensor1);                      // [240, 240, 1]

  // Put uploaded image in channel 0 (e.g. T1), blanks elsewhere
  const stacked = tfjs.stack([tensor1, tensor1, tensor1, tensor1], -1).reshape([1, 240, 240, 4]) as tfjs.Tensor4D;
  // const stacked = tfjs.stack([blank, blank, blank, tensor1], -1).reshape([1, 240, 240, 4]) as tfjs.Tensor4D;


  return { tensor: stacked, metadata: { dims: [240, 240, 4] } };
};

export const overlaySegmentation = async (mask: tfjs.Tensor2D): Promise<string> => {
  console.log('Mask shape:', mask.shape);         // should be [240, 240] or [240, 240, 1]

  const maskArray = await mask.array();
  console.log('Unique labels in mask:', new Set(maskArray.flat()));

  const hasTumor = maskArray.flat().some(label => label !== 0);
  console.log('Tumor present in mask?', hasTumor);


  const canvas = createCanvas(240, 240);
  const ctx = canvas.getContext('2d');
  const overlay = ctx.createImageData(240, 240);

  for (let y = 0; y < 240; y++) {
    for (let x = 0; x < 240; x++) {
      const label = maskArray[y][x];
      const i = (y * 240 + x) * 4;
      let [r, g, b] = [0, 0, 0];

      if (label === 1) [r, g, b] = [255, 0, 0];       // NCR - red
      if (label === 2) [r, g, b] = [0, 255, 0];       // ED - green
      if (label === 3) [r, g, b] = [0, 0, 255];       // ET - blue

      overlay.data[i] = r;
      overlay.data[i + 1] = g;
      overlay.data[i + 2] = b;
      overlay.data[i + 3] = label ? 120 : 0;          // alpha channel
    }
  }

  ctx.putImageData(overlay, 0, 0);
  return canvas.toDataURL();
};