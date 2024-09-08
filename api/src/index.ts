import express, { Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import dotenv from 'dotenv';
dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

const upload = multer({ dest: 'uploads/' }); // create 'uploads' folder

app.use(express.json()); // parse 'application/json' content-type

// Handle file upload 
app.get('/upload', upload.single('image'), (req: Request, res: Response) => {
  const filePath = req.file?.path;
  if (filePath) res.send({ message: 'File uploaded successfully!', filePath });
  else res.status(400).send({ message: 'File upload failed.' }); // bad client request
});

// Serve static files
app.listen(port, () => console.log(`Server is running on http://localhost:${port}`));