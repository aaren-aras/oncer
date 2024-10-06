import { Request, Response } from 'express';
import { predict } from '../services/model.service.ts';
import fs from 'fs';

export const predictTumour = async (req: Request, res: Response) => {
  const imagePath = req.file?.path;
  if (!imagePath) return res.status(400).json({ error: 'No file uploaded' });

  try {
    const result = await predict(imagePath);
    res.json({ prediction: result });
  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).json({ error: 'Prediction failed' });
  } finally {
    fs.unlinkSync(imagePath); // clean up file after prediction
  }
};