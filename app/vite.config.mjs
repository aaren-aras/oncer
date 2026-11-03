import { defineConfig, loadEnv } from 'vite' // autocompletion and type checking for this file
import { fileURLToPath, URL } from 'node:url'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, fileURLToPath(new URL('..', import.meta.url)), ''); // point to root dir
  const appPort = env.APP_PORT || '5173';
  const apiPort = env.API_PORT || '5000';

  return {
    css: {
      preprocessorOptions: {
        scss: {
          api: 'modern-compiler' // use dart-sass
        }
      }
    },
    plugins: [
      vue(),
    ],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    server: {
      port: Number(appPort), // frontend dev server port
      proxy: {
        '/api': {
          target: `http://localhost:${apiPort}`, // route frontend requests to FastAPI backend address 
          changeOrigin: true, // tell backend that requests are coming from apiPort, not appPort 
        }
      }
    }
  }
})