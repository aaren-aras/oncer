<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>

<template>
  <div class="greetings">
    <h1 class="green">{{ title }}</h1>
    <h4>
      <u>O</u>NCER Neural Classificaton / Evaluation Reporter
      (or... <u>ON</u>OLOGY + CAN<u>CER</u>)
    </h4>
    <h3>
      An all-in-ONE solution for tumour detection with a SINGLE upload!
      Powered by


      <a href="https://vuejs.org/" target="_blank" rel="noopener">Vue 3</a> +
      <a href="" target="_blank" rel="noopener">Express</a> + 
      <a href="" target="_blank" rel="noopener">TensorFlow</a>. 
    </h3>
  </div>
</template>

<style scoped lang="scss">
  @use '../../assets/scss/global.scss' as *;

  h1 {
    font-weight: 500;
    font-size: 2.6rem;
    color: $accent-3;
    padding: 0;
    margin: 0;
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
    position: relative;
    top: -10px;
  }

  h4 {
    width: 70%;
    font-size: 1rem;
    font-weight: normal;
    margin-block-start: 0;
    margin-block-end: 0.5rem; // tweak this to taste
  }

  h3 {
    width: 70%;
    font-size: 1.2rem;
    font-weight: normal;
  }

  .greetings h1,
  .greetings h3 {
    text-align: center;
  }

  @media (min-width: 1024px) {
    .greetings h1,
    .greetings h3 {
      text-align: left;
    }
  }
</style>
