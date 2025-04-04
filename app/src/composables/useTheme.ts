import { ref, onMounted } from 'vue' // auto-updates wherever it's used (App.vue) + runs after component is added to DOM

type Theme = 'light' | 'dark'
const theme = ref<Theme>('light')

export function useTheme() {
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)') // returns true if user has dark mode enabled

  const setTheme = (value: Theme) => {
    theme.value = value
    const classList = document.documentElement.classList
    classList.remove('light-mode', 'dark-mode')
    classList.add(`${value}-mode`)
    localStorage.setItem('theme', value)
  }

  const toggleTheme = () => { 
    setTheme(theme.value === 'dark' ? 'light' : 'dark')
  }

  onMounted(() => {
    const savedTheme = localStorage.getItem('theme') as Theme | null
    if (savedTheme === 'light' || savedTheme === 'dark') setTheme(savedTheme)
    else setTheme(prefersDark.matches ? 'dark' : 'light')

    prefersDark.addEventListener('change', (e) => {
      if (!localStorage.getItem('theme')) setTheme(e.matches ? 'dark' : 'light')
    })
  })

  return { theme, toggleTheme }
}
