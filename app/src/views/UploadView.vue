<template>
  <div class="about">
    <h1>Upload here:</h1>
    <input type="file" @change="uploadFile" accept="image/*" />
    <div v-show="message">{{ message }}</div>
  </div>
</template>

<script setup lang="ts">
  import { ref } from 'vue';

  const message = ref(''); // make reactive
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
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      message.value = data.message;
    } catch (e) {
      console.error('Error uploading file:', e);
      message.value = 'File upload failed.';
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
</style>
