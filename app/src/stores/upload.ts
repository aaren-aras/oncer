import { defineStore } from 'pinia';

export const useUploadStore = defineStore('upload', {
  state: () => ({
    uploads: [] as { image: string; filename: string; prediction: string }[],
  }),
  actions: {
    addUpload(image: string, filename: string, prediction: string) {
      this.uploads.push({ image, filename, prediction });
      localStorage.setItem('uploads', JSON.stringify(this.uploads)); // persist across reloads
    },
    loadFromLocalStorage() {
      const storedState = localStorage.getItem('uploads');
      if (storedState) this.uploads = JSON.parse(storedState);
    }
  }
});
