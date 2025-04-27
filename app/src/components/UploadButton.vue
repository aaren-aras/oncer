<template>
  <form id="upload-form" @submit.prevent>
    <input type="file" id="image" @change="handleFileUpload" accept="image/*" required hidden />
    <label for="image" class="custom-file-label">Upload Image</label>
  </form>
</template>

<script setup lang="ts">
  import { useRouter } from 'vue-router';
  import { useUploadStore } from '@/stores/upload';

  const router = useRouter()
  const store = useUploadStore(); // use Pinia store for state management
  // const emit = defineEmits(['image-uploaded']); // define emit event

  const handleFileUpload = async (e: Event) => {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    const formData = new FormData(); // send file as FormData object
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      // const imageUrl = URL.createObjectURL(file); // create (temp.) preview URL for display
      const base64Image = await convertToBase64(file);

      store.addUpload(base64Image, file.name, data.prediction); // add to Pinia store
      router.push('/upload'); // Redirect to upload page after upload

      // emit('image-uploaded', {
      //     prediction: data.prediction,
      //     image: imageUrl,
      //     originalName: file.name,
      //   });

    } catch (e) {
      console.error('Error uploading file:', e);
    }
  };

  /* Convert img to Base64 to persist in localStorage (unlike temp. blob URLs) */
  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string); 
      reader.onerror = err => reject(err); 
      reader.readAsDataURL(file); 
    })
  }
</script>

<style scoped lang="scss">
    @use '../../assets/scss/global.scss' as *;

    form {
      display: inline-block;
    }
    
    input {
    // display: block;
    // margin: 0 auto;
    padding: 10px;
    font-size: 16px;
    border-radius: 5px;
    border: 1px solid var(--color-border);
    background-color: var(--color-background);
    color: var(--color-text);
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
    padding: 0.5rem 1rem;
    background-color: var(--color-text);
    color: #fff;
    border-radius: 8px;
    cursor: pointer;
    @include transition-ease;
  }

  .custom-file-label:hover {
    background-color: #374151;
  }
</style>