<!-- <template>
  <form id="upload-form" @submit.prevent>
    <input type="file" id="image" @change="handleFileUpload" accept=".png, .jpg, .jpeg, .nii, .nii.gz" required hidden />
    <label for="image" class="custom-file-label">Upload File</label>
  </form>
</template> -->
<template>
  <form id="upload-form" @submit.prevent="handleSubmit">
    <div class="modality-uploads">
      <div>
        <input type="file" id="t1" @change="(e) => handleFileChange('t1')(e)" accept=".png, .jpg, .jpeg" required hidden />
        <label for="t1" class="custom-file-label" :class="{'has-file': t1File}">T1</label>
      </div>
      <div>
        <input type="file" id="t2" @change="(e) => handleFileChange('t2')(e)" accept=".png, .jpg, .jpeg" required hidden />
        <label for="t2" class="custom-file-label" :class="{'has-file': t2File}">T2</label>
      </div>
      <div>
        <input type="file" id="t1ce" @change="(e) => handleFileChange('t1ce')(e)" accept=".png, .jpg, .jpeg" required hidden />
        <label for="t1ce" class="custom-file-label" :class="{'has-file': t1ceFile}">T1CE</label>
      </div>
      <div>
        <input type="file" id="flair" @change="(e) => handleFileChange('flair')(e)" accept=".png, .jpg, .jpeg" required hidden />
        <label for="flair" class="custom-file-label" :class="{'has-file': flairFile}">FLAIR</label>
      </div>
    </div>
    <button type="submit">UPLOAD</button>
  </form>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useUploadStore } from '@/stores/upload';

const router = useRouter();
const store = useUploadStore();

const t1File = ref<File | null>(null);
const t1ceFile = ref<File | null>(null);
const t2File = ref<File | null>(null);
const flairFile = ref<File | null>(null);

const handleFileChange = (modality: string) => (e: Event) => {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;

  console.log(`Selected file for ${modality}:`, file); 


  if (!file) return;

  switch (modality) {
    case 't1': t1File.value = file; break;
    case 't1ce': t1ceFile.value = file; break;
    case 't2': t2File.value = file; break;
    case 'flair': flairFile.value = file; break;
  }
};

const handleSubmit = async () => {
  if (!t1File.value || !t1ceFile.value || !t2File.value || !flairFile.value) {
    alert('Please upload all four modalities before submitting.');
    return;
  }

  const formData = new FormData();
  formData.append('t1', t1File.value);
  formData.append('t1ce', t1ceFile.value);
  formData.append('t2', t2File.value);
  formData.append('flair', flairFile.value);

  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();

    // For preview, convert one of the files (say flair) to base64
    const base64Image = await convertToBase64(t1ceFile.value);

    store.addUpload(base64Image, flairFile.value.name, data.prediction, data.overlay);
    router.push('/upload');
  } catch (e) {
    console.error('Error uploading files:', e);
  }
};

const convertToBase64 = (file: File): Promise<string> => {
  /* Converts uploads to Base64 to persist in localStorage (unlike blob URLs) */
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = err => reject(err);
    reader.readAsDataURL(file);
  });
};



  // import { useRouter } from 'vue-router';
  // import { useUploadStore } from '@/stores/upload';

  // const router = useRouter()
  // const store = useUploadStore(); // use Pinia store for state management
  // // const emit = defineEmits(['image-uploaded']); // define emit event

  // const handleFileUpload = async (e: Event) => {
  //   const input = e.target as HTMLInputElement;
  //   const file = input.files?.[0];
  //   if (!file) return;

  //   const formData = new FormData(); // send file as FormData object
  //   // formData.append('file', file);

  //   formData.append('t1', t1File);
  //   formData.append('t1ce', t1ceFile);
  //   formData.append('t2', t2File);
  //   formData.append('flair', flairFile);

  //   try {
  //     const response = await fetch('http://localhost:5000/predict', {
  //       method: 'POST',
  //       body: formData,
  //     });

  //     if (!response.ok) {
  //       throw new Error('Network response was not ok');
  //     }

  //     const data = await response.json();
  //     // const imageUrl = URL.createObjectURL(file); // create (temp.) preview URL for display
  //     const base64Image = await convertToBase64(file);

  //     store.addUpload(base64Image, file.name, data.prediction, data.overlay); // add to Pinia store
  //     router.push('/upload'); // Redirect to upload page after upload

  //     // emit('image-uploaded', {
  //     //     prediction: data.prediction,
  //     //     image: imageUrl,
  //     //     originalName: file.name,
  //     //   });

  //   } catch (e) {
  //     console.error('Error uploading file:', e);
  //   }
  // };

  // /* Convert img to Base64 to persist in localStorage (unlike temp. blob URLs) */
  // const convertToBase64 = (file: File): Promise<string> => {
  //   return new Promise((resolve, reject) => {
  //     const reader = new FileReader();
  //     reader.onload = () => resolve(reader.result as string); 
  //     reader.onerror = err => reject(err); 
  //     reader.readAsDataURL(file); 
  //   })
  // }
</script>

<style scoped lang="scss">
  @use '../../assets/scss/global.scss' as *;

  form {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: -0.75rem;
    margin-left: 1rem;
    
    .modality-uploads {
      // display: grid;
      // grid-template-columns: repeat(2, 1fr);
      // grid-template-columns: max-content 1fr;

      display: flex;
      gap: 10px;

      input {
      // display: block;
      margin: 0;
      // padding: 10px;
      font-size: 16px;
      // border-radius: 5px;
      // border: 1px solid var(--color-border);
  
      cursor: pointer;

      &:focus {
        outline: none;
        border-color: var(--color-heading);
        box-shadow: 0 0 0 2px rgba(100, 150, 255, 0.4);
      }

      &::placeholder {
        color: rgba(var(--color-text), 0.5); // If var doesn't work here, use a fallback color
      }
    }

      .custom-file-label {
        display: inline-block;
        padding: 0.5rem 0.5rem;
        background-color: var(--accent-2);
        color: var(--secondary);
        border-radius: 4px;
        cursor: pointer;
        @include transition-ease;

        &:hover {
          background-color: #374151;
        }

        &.has-file {
          background-color: #91daab; // green to show it's filled
        }
      }

    }

    
    button {
      height: fit-content;
      font-size: 1.1rem;
      font-weight: 500;
      background-color: $accent-3;
      color: var(--primary);
      border: 0;
      border-radius: 4px;
      padding: 16px;
   
      cursor: pointer;
      @include transition-ease;

      &:hover {
        background-color: var(--accent-2);
      }
    }
  } 
</style>