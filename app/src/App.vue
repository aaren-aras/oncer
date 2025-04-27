
<template>
  <header>
    <img alt="Oncer logo" class="logo" src="../assets/logo.svg" width="150" height="150" />

    <div class="wrapper">
      <WelcomeText title="Oncer" />
      <nav>
        <RouterLink to="/">Welcome</RouterLink>
        <UploadButton @image-uploaded="handleImageUpload" />
        <!-- <RouterLink to="/upload">Upload Scan</RouterLink> -->
      </nav>
    </div>

    <button @click="toggleTheme" class="theme-toggle">
      Toggle to {{ theme === 'dark' ? 'Light' : 'Dark' }} Mode
    </button>
  </header>

  <main>
    <RouterView />
  </main>
</template>

<script setup lang="ts">
  import { RouterLink, RouterView } from 'vue-router'
  import { useTheme } from '@/composables/useTheme'
  import WelcomeText from './components/WelcomeText.vue'
  import UploadButton from './components/UploadButton.vue'

  const { theme, toggleTheme } = useTheme()


  import { ref } from 'vue';
  
  const uploadedImage = ref<string | null>(null);
  const predictionResult = ref<string>('');
  const handleImageUpload = (data: { image: string; prediction: string; originalName: string }) => {
    uploadedImage.value = data.image;
    predictionResult.value = `Tumor ${data.prediction.toLowerCase()}`; 
  };

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
  line-height: 1.5;
  max-height: 100vh;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

nav {
  // width: 100%;
  font-size: 12px;
  text-align: center;
  margin-top: 2rem;
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
