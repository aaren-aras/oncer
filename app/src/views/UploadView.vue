<template>
  <h1>Upload here:</h1>
  <form id="upload-form" @submit.prevent>
    <input type="file" id="image" @change="uploadFile" accept="image/*" required/>
  </form>
  <div v-show="message">{{ message }}</div>
  <div v-show="result">{{ result }}</div>
</template>

<script setup lang="ts">
  import { ref } from 'vue';

  const message = ref(''); // make reactive
  const result = ref('');

  const uploadFile = async (e: Event) => {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      message.value = 'No file selected.';
      return;
    }

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
      message.value = 'Upload successful!'; // Update message on success
      result.value = data.prediction; // Set to prediction result    
    } catch (e) {
      console.error('Error uploading file:', e);
      message.value = 'File upload failed.';
      result.value = ''; // Clear result on error
    }
  };
</script>

<style>
  @media (min-width: 1024px) {
    .about {
      min-height: 100vh;
      display: flex;
      align-items: center;
    }
  }

  #result {
    font-weight: bold;
    margin-top: 20px;
    color: green;
  }

  #message {
    color: red;
  }
</style>
