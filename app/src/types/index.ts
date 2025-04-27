export interface UploadResult {
  image: string; // preview URL (from URL.createObjectURL)
  filename: string; // original filename
  prediction: string; // from backend (currently "Positive" or "Negative")
}