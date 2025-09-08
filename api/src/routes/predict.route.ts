import { Router } from 'express';
import { upload } from '../middleware/multer.ts';
import { predictTumour } from '../controllers/predict.controller.ts';

const router = Router();

// Upload endpoint for any format (.jpg, .png, .dcm, .nii)
// router.post('/', upload.single('file'), predictTumour);

router.post('/', upload.fields([
  { name: 't1', maxCount: 1 },
  { name: 't1ce', maxCount: 1 },
  { name: 't2', maxCount: 1 },
  { name: 'flair', maxCount: 1 }
]), predictTumour);

export default router;