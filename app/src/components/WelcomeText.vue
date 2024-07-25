<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>

<template>
  <div class="greetings">
    <h1 class="green">{{ title }}</h1>
    <h3>
      An all-in-one solution for cancer detection with a mere click of a button!
      Powered by
      <a href="https://vuejs.org/" target="_blank" rel="noopener">Vue 3</a> +
      <a href="" target="_blank" rel="noopener">Express</a> + 
      <a href="" target="_blank" rel="noopener">???</a>. 
    </h3>
  </div>
</template>

<style scoped>
h1 {
  font-weight: 500;
  font-size: 2.6rem;
  position: relative;
  top: -10px;
}

h3 {
  font-size: 1.2rem;
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
