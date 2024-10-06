import express from 'express';
import cors from 'cors';
// import dotenv from 'dotenv';
// dotenv.config();

import predictRoute from './routes/predict.route.ts';
import { loadModel } from './services/model.service.ts';

const app = express();

app.use(cors()); // enable CORS for all routes
app.use('/models', express.static('models'));
app.use('/predict', predictRoute);

loadModel() // load before listening

const port = 5000;
app.listen(port, () => console.log(`Server is running on http://localhost:${port}`));

export default app;