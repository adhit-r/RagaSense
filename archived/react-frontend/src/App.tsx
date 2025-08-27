import { useState } from 'lynx'
import { Router, Route } from '@lynx/router'
import { ThemeProvider } from '@/components/ThemeProvider'
import { Navbar } from '@/components/Navbar'
import { Home } from '@/pages/Home'
import { RagaDetector } from '@/pages/RagaDetector'
import { RagaList } from '@/pages/RagaList'
import { Toaster } from '@/components/Toaster'

export function App() {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system')

  return (
    <ThemeProvider theme={theme} onThemeChange={setTheme}>
      <div className="min-h-screen bg-background text-foreground">
        <Navbar theme={theme} onThemeChange={setTheme} />
        <main className="container mx-auto px-4 py-8">
          <Router>
            <Route path="/" component={Home} />
            <Route path="/detect" component={RagaDetector} />
            <Route path="/ragas" component={RagaList} />
          </Router>
        </main>
        <Toaster />
      </div>
    </ThemeProvider>
  )
}
