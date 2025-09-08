export interface UploadResult {
  image: string; // preview URL (from URL.createObjectURL)
  filename: string; // original filename
  prediction: string; // from backend (currently 'Tumour(s) detected' and 'No tumour detected')
  overlay: string; // from backend (Base64 PNG overlay for segmentation mask)
}