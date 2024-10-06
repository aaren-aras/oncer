import { Router } from 'express';
import { upload } from '../middleware/multer.ts';
import { predictTumour } from '../controllers/predict.controller.ts';

const router = Router();

router.post('/', upload.single('image'), predictTumour);

export default router;