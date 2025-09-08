<template>
  <section v-if="store.isEmpty" id="empty-upload-view">
    <p>No images have been uploaded yet.</p>
  </section>
  <section v-else id="upload-view">
    <div class="selected-view">
      <div class="img-wrapper" ref="imageWrapper">
        <img :src="selectedImage.image" ref="baseImage" @load="drawOverlay" />
        <canvas class="overlay-canvas" ref="overlayCanvas"></canvas>
      </div>
      <!-- <img v-if="selectedImage" :src="selectedImage.image" /> -->
      <p>{{ selectedImage?.prediction }}</p> 
    </div>

    <!-- <div v-if="isStoreEmpty" class="">
      <p>No images have been uploaded yet.</p>
    </div> -->

    <div class="thumbnail-row">
      <div
        v-for="upload in otherImages"
        :key="upload.filename"
        class="thumbnail"
        @click="selectImage(upload)"
      >
        <img :src="upload.image" alt=""/>
      </div>
    </div>

    <div class="reset-wrapper">
      <button class="reset" @click="store.resetUploads()">RESET</button>
    </div>
  </section>
</template>

<script setup lang="ts">
  import { ref, computed, watch, onMounted, nextTick } from 'vue';
  import { useUploadStore } from '@/stores/upload';
  import type { UploadResult } from '@/types';
  
  const store = useUploadStore();
  const selectedImage = ref<UploadResult | null>(null);

  const otherImages = computed(() => {
    return store.uploads.filter((upload) => upload !== selectedImage.value);
  })

  watch(
    () => store.uploads.length, // run callback when number of stored uploads changes
    () => {
      selectedImage.value = store.uploads[0] ?? null;
    },
    { immediate: true } // initialize selectedImage immediately after refreshes (on mount)
  );

  const selectImage = (upload: UploadResult) => {
    selectedImage.value = upload;
  }

  const baseImage = ref<HTMLImageElement | null>(null);
  const overlayCanvas = ref<HTMLImageElement | null>(null);

  const drawOverlay = async() => {
    await nextTick();
    if (!selectedImage.value?.overlay || !overlayCanvas.value || !baseImage.value) return;

      const img = baseImage.value;
      const canvas = overlayCanvas.value;

      // Match canvas size to image size for perfect overlay
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Clear previous overlay
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Create image object for overlay
      const overlayImg = new Image();
      overlayImg.src = selectedImage.value.overlay;

      overlayImg.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 0.5; // semi-transparent overlay
        ctx.drawImage(overlayImg, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
      };
  }

  onMounted(() => {
    if (baseImage.value) {
      baseImage.value.addEventListener('load', drawOverlay);
    }
  });

  // const uploads = storeToRefs(store).uploads;
  // import UploadButton from '@/components/UploadButton.vue';  
  // const uploadedImage = ref<string | null>(null);
  // const predictionResult = ref<string>('');
  // const handleImageUpload = (data: { image: string; prediction: string; originalName: string }) => {
  //   uploadedImage.value = data.image;
  //   predictionResult.value = `Tumor ${data.prediction.toLowerCase()}`; 
  // };

</script>

<style scoped lang="scss">
  @use '../../assets/scss/global.scss' as *;
  // @use '../../assets/scss/palette';

  #empty-upload-view {
    text-align: center;
    font-size: 1.5rem;
  }

  #upload-view {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
    width: 100%;

    .selected-view {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1em;
      // border: 1px solid red;

      .img-wrapper {
        position: relative;
        display: inline-block;
        max-width: 50em;

        img {
          // width: 100%;
          //   height: auto;
          //   display: block;

          display: block;
          // width: 50rem;
          // border: 1px solid var(--color-border);
          border: 2px ridge $accent-3;
          border-radius: 5px;
          // color: palette.$accent-3;
        }

        .overlay-canvas {
          // width: 100%;
          //   height: auto;
          //   display: block;


          border-radius: 5px;
          position: absolute;
          top: 0;
          left: 0;
          z-index: 100;
          pointer-events: none; /* so mouse clicks pass through */
          user-select: none;
        }
      }

      p {
        font-size: 1.25rem;
        font-weight: bold;
      }
    }

    .thumbnail-row {
      display: flex;
      justify-content: flex-start;
      gap: 1rem;
      width: 30rem;
      padding: 10px;
      border-top: 1px solid var(--color-border);
      overflow-x: auto;


      .thumbnail {
        flex: 0 0 auto;
        width: 10rem;
        height: 10rem;
        margin-top: 1rem;
        cursor: pointer;

        img {
          object-fit: cover;
          border-radius: 20px;
          @include transition-ease;

          &:hover {
            transform: scale(1.05);
          }
        }
      }
    }

    .reset-wrapper {
      .reset {
        // background-color: var(--);
        background-color: $accent-3;
        color: #fff;
        border: 1px solid var(--color-border);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;  
        @include transition-ease;
        
        &:hover {
          opacity: 0.6;
        }
      }
    }
  }
</style>
