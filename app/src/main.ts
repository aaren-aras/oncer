import '../assets/scss/global.scss'; 

import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { useUploadStore } from './stores/upload';

import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')

const uploadStore = useUploadStore();
uploadStore.loadFromLocalStorage(); // load uploads from localStorage on app start
