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
      <div class="prediction-row">
        <p>{{ selectedImage?.prediction }}</p>

        <div class="legend">
          <div class="legend-item">
            <span class="legend-swatch" style="--legend-alpha: 0.50"></span>
            <span class="legend-label">NCR</span>
          </div>
          <div class="legend-item">
            <span class="legend-swatch" style="--legend-alpha: 0.71"></span>
            <span class="legend-label">ED</span>
          </div>
          <div class="legend-item">
            <span class="legend-swatch" style="--legend-alpha: 1"></span>
            <span class="legend-label">ET</span>
          </div>
        </div>
      </div>
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
  import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue';
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

      // Match canvas size to image's rendered size (NOT natural pixel size)
      // Overlay dims stay consistent across diff source resolutions
      const { width, height } = img.getBoundingClientRect();

      canvas.width = width;
      canvas.height = height;

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

  // Re-draw if the window/container resizes, so the canvas stays matched to the image
  const handleResize = () => drawOverlay();

  onMounted(() => {
    if (baseImage.value) {
      baseImage.value.addEventListener('load', drawOverlay);
    }
    window.addEventListener('resize', handleResize);
  });

  onUnmounted(() => {
    window.removeEventListener('resize', handleResize);
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
          display: block;
          max-width: 100%;
          height: auto;
          border: 2px ridge $accent-3;
          border-radius: 5px;
        }

        .overlay-canvas {
          width: 100%;
          height: 100%;
          border-radius: 5px;
          position: absolute;
          top: 0;
          left: 0;
          z-index: 100;
          pointer-events: none; // mouse clicks pass through
          user-select: none;
        }
      }

      p {
        font-size: 1.25rem;
        font-weight: bold;
      }
    }

    .prediction-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1.5rem;
      flex-wrap: wrap;

      p {
        font-size: 1.25rem;
        color: var(--color-heading);
        font-weight: 500;
        margin: 0;
      }

      .legend {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        align-items: center;

        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.4rem;
          font-size: 0.9rem;
        }

        .legend-swatch {
          display: inline-block;
          width: 1rem;
          height: 1rem;
          border-radius: 4px;
          border: 1px solid var(--color-border);
          background-color: rgba(209, 82, 255, var(--legend-alpha));
        }

        .legend-label {
          color: var(--color-text);
        }
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
