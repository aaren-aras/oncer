import multer from 'multer';

// TO DO: upload files to AWS S3 bucket
export const upload = multer({ dest: 'uploads/' }); 