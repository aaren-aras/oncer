
<template>
  <header>
    <img alt="ONCER Logo" class="logo" src="../assets/logo-new.svg" />

    <div class="wrapper">
      <LeftPanel title="ONCER&nbsp; 𖡎" />
      <nav>
        <UploadButton />
        <div class="router-links">
          <RouterLink to="/">Welcome</RouterLink>
          <RouterLink to="/upload">Uploads</RouterLink>
        </div>
      </nav>
    </div>

    <button @click="toggleTheme" class="theme-toggle">
      {{ theme === 'dark' ? 'Light Mode ☀️' : 'Dark Mode 🌙' }}
    </button>
  </header>
  <footer>
    <p>
      &copy; {{ new Date().getFullYear() }} Aaren Arasaratnam
      &nbsp;&bull;&nbsp; v0.1.0
      &nbsp;&bull;&nbsp; Released under the
      <a href="https://github.com/aaren-aras/oncer/blob/main/LICENSE" target="_blank" rel="noopener">MIT License</a>
    </p>
    <!-- <p class="data-attribution">
      Trained on the
        <a href="https://www.synapse.org/#!Synapse:syn25829067" target="_blank" rel="noopener">RSNA-ASNR-MICCAI BraTS 2021</a>
      dataset. Not for clinical or commercial use.
    </p> -->
  </footer>

  <main>
    <RouterView />
  </main>
</template>

<script setup lang="ts">
  import { RouterLink, RouterView } from 'vue-router'
  import { useTheme } from '@/composables/useTheme'
  import LeftPanel from './components/LeftPanel.vue'
  import UploadButton from './components/UploadButton.vue'

  const { theme, toggleTheme } = useTheme()


  // import { ref } from 'vue';
  
  // const uploadedImage = ref<string | null>(null);
  // const predictionResult = ref<string>('');
  // const handleImageUpload = (data: { image: string; prediction: string; originalName: string }) => {
  //   uploadedImage.value = data.image;
  //   predictionResult.value = `Tumor ${data.prediction.toLowerCase()}`; 
  // };

</script>


<style scoped lang="scss">
  @use '../assets/scss/global.scss' as *;

  main {
    // width: 100%;
    // max-width: 100rem;
    // max-width: 100%;
    min-width: 100%;
    // margin: 0 auto;
    // padding: 2rem 1rem;
  }

  .theme-toggle {
    background-color: transparent;
    color: var(--color-text);
    border: 1px solid var(--color-border);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    position: absolute;
    bottom: 30px;
    left: 30px;
    cursor: pointer;
    @include transition-ease;

    &:hover {
      opacity: 0.5;
    }
  }

header {
  display: flex;
  // gap: 0.5rem;
  line-height: 1.5;
  // max-height: 100vh;
}

img {
  display: inline-block;
  width: 10rem;
  height: 10rem;
  // margin-right: 1rem;
}

nav {
  display: flex;
  flex-direction: column;
  // width: 100%;
  // font-size: 12px;
  text-align: center;
  // margin-top: 2rem;

  .router-links {
    font-size: 1.1rem;
    margin-top: 1.5rem;
  }


}

nav a.router-link-exact-active {
  // color: var(--color-text);
  color: $accent-3;
  // color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  color: var(--color-text);
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}

footer {
  // background-color: transparent;
  color: var(--color-text);
  // border: 1px solid var(--color-border);
  // padding: 0.5rem 1rem;
  // border-radius: 8px;
  display: flex;
  position: absolute;
  bottom: 30px;
  right: 30px;

  footer a {
    cursor: pointer;
    @include transition-ease;
  }
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }

  nav {
    text-align: left;
    margin-left: -1rem;
    font-size: 1rem;

    padding: 1rem 0;
    margin-top: 1rem;
  }
}
</style>
