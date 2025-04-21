import { defineStore } from 'pinia';
import { ref } from 'vue';
import type { UploadResult } from '@/types';

// export const useUploadStore = defineStore('upload',  {
//   state: () => ({
//     uploads: [] as { image: string; filename: string; prediction: string }[],
//   }),
//   actions: {
//     addUpload(image: string, filename: string, prediction: string) {
//       // this.uploads.push({ image, filename, prediction });
//       this.uploads.unshift({ image, filename, prediction }); // add to start of array instead of end (push)
//       localStorage.setItem('uploads', JSON.stringify(this.uploads)); // persist across reloads
//     },
//     loadFromLocalStorage() {
//       const storedState = localStorage.getItem('uploads');
//       if (storedState) this.uploads = JSON.parse(storedState);
//     }
//   }
// });

export const useUploadStore = defineStore('upload', () => {
  // const uploads = ref<{ image: string, filename: string, prediction: string }[]>([]);
  const uploads = ref<UploadResult[]>([]);

  const addUpload = (image: string, filename: string, prediction: string) => { 
    uploads.value.unshift({ image, filename, prediction }); // add to start of array instead of end (push)
    localStorage.setItem('uploads', JSON.stringify(uploads.value)); // persist across reloads
  }

  const resetUploads = () => {
    uploads.value = [];
    localStorage.removeItem('uploads');
  }

  const loadFromLocalStorage = () => {
    const storedState = localStorage.getItem('uploads');
    if (storedState) uploads.value = JSON.parse(storedState);
  }

  return { uploads, addUpload, resetUploads, loadFromLocalStorage };
})