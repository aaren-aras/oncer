import { Request, Response } from 'express';
import { predict } from '../services/model.service.ts';
import { execSync } from 'child_process';
import fs from 'fs';
// import path from 'path';

// export const predictTumour = async (req: Request, res: Response) => {
//   const imagePath = req.file?.path;
//   if (!imagePath) return res.status(400).json({ error: 'No file uploaded' });

//   try {
//     const result = await predict(imagePath);
//     res.json({ prediction: result });
//   } catch (error) {
//     console.error('Error during prediction:', error);
//     res.status(500).json({ error: 'Prediction failed' });
//   } finally {
//     fs.unlinkSync(imagePath); // clean up file after prediction
//   }
// };

export const predictTumour = async (req: Request, res: Response) => {
  const files = req.files as {
    [fieldname: string]: Express.Multer.File[];
  };

  if (!files.t1 || !files.t1ce || !files.t2 || !files.flair) {
    return res.status(400).json({ error: 'All four modalities must be uploaded (t1, t1ce, t2, flair)' });
  }

  try {
    const paths = {
      t1: files.t1[0].path,
      t1ce: files.t1ce[0].path,
      t2: files.t2[0].path,
      flair: files.flair[0].path
    };

    const result = await predict(paths); // send all 4
    res.json(result);
  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).json({ error: 'Prediction failed' });
  } finally {
    // Cleanup all 4 files
    Object.values(files).forEach(fileArr =>
      fileArr.forEach(file => fs.unlinkSync(file.path))
    );
  }
};

// export const predictTumour = async (req: Request, res: Response) => {
//   const filePath = req.file?.path;
//   const mimetype = req.file?.mimetype;

//   if (!filePath || !mimetype) return res.status(400).json({ error: 'No file uploaded' });

//   const npyPath = filePath.replace(/\.(png|jpg|jpeg)$/i, '.npy');
//   if (mimetype.includes('image')) {
//     try {
//       execSync(`python src/scripts/png_to_npy.py ${filePath} ${npyPath}`);
//     } catch (e) {
//       console.error('Conversion to .npy failed:', e);
//       return res.status(500).json({ error: 'Failed to convert image to NPY' });
//     }
//   }


//   try {
//     const result = await predict(filePath, mimetype);
//     res.json(result);
//   } catch (error) {
//     console.error('Error during prediction:', error);
//     res.status(500).json({ error: 'Prediction failed' });
//   } finally {
//     fs.unlinkSync(filePath); // Clean up uploaded file
//   }
// };


/*const ext = path.extname(imagePath);
if (ext === '.nii' || ext === '.nii.gz') {
  // Process full NIfTI
  const result = await predictFromNifti(imagePath);
  return res.json(result);
} else if (ext === '.png' || ext === '.jpg' || ext === '.jpeg') {
  const result = await predictFromImage(imagePath); // handles single slice
  return res.json(result);
} */ /* else if (ext === '.dcm') {
  const imgPath = await convertDicomToImage(imagePath); // convert to PNG
  const result = await predictFromImage(imgPath);
  return res.json(result);
} */
