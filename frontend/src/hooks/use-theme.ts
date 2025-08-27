import * as React from "react"
import { useTheme as useNextTheme } from "next-themes"

export function useTheme() {
  const { theme, setTheme, systemTheme } = useNextTheme()
  const [mounted, setMounted] = React.useState(false)

  React.useEffect(() => {
    setMounted(true)
  }, [])

  const currentTheme = theme === 'system' ? systemTheme : theme

  return {
    theme: currentTheme,
    setTheme,
    isDark: currentTheme === 'dark',
    mounted,
  }
}
