import { Link } from '@lynx/router'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/Card'
import { Button } from '@/components/Button'

export function Home() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="text-center space-y-6 py-12">
        <div className="space-y-4">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
            Discover the Soul of
            <span className="text-primary block">Indian Classical Music</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Upload audio files and instantly identify the raga using advanced AI. 
            Explore the rich tradition of Indian classical music through our comprehensive database.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/detect">
            <Button size="lg" className="text-lg px-8 py-4">
              Detect Raga
            </Button>
          </Link>
          <Link href="/ragas">
            <Button variant="outline" size="lg" className="text-lg px-8 py-4">
              Browse Ragas
            </Button>
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section className="grid md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>AI-Powered Detection</CardTitle>
            <CardDescription>
              Advanced machine learning algorithms analyze audio patterns to identify ragas with high accuracy.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">Real-time analysis</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">Multiple audio formats</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">High accuracy results</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Comprehensive Database</CardTitle>
            <CardDescription>
              Access detailed information about thousands of ragas from both Hindustani and Carnatic traditions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span className="text-sm">5,893+ ragas</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span className="text-sm">Scale patterns</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span className="text-sm">Cultural context</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Educational Resources</CardTitle>
            <CardDescription>
              Learn about ragas, their characteristics, and the cultural significance behind each musical tradition.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm">Raga analysis</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm">Historical context</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm">Performance guides</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* How It Works Section */}
      <section className="space-y-6">
        <div className="text-center">
          <h2 className="text-3xl font-bold">How It Works</h2>
          <p className="text-muted-foreground mt-2">
            Simple steps to discover the raga in your music
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto">
              <span className="text-primary-foreground font-bold text-xl">1</span>
            </div>
            <h3 className="text-xl font-semibold">Upload Audio</h3>
            <p className="text-muted-foreground">
              Upload your audio file in WAV, MP3, OGG, FLAC, or M4A format. 
              Our system supports files up to 50MB.
            </p>
          </div>

          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto">
              <span className="text-primary-foreground font-bold text-xl">2</span>
            </div>
            <h3 className="text-xl font-semibold">AI Analysis</h3>
            <p className="text-muted-foreground">
              Our advanced AI analyzes the audio patterns, pitch distribution, 
              and musical characteristics to identify the raga.
            </p>
          </div>

          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto">
              <span className="text-primary-foreground font-bold text-xl">3</span>
            </div>
            <h3 className="text-xl font-semibold">Get Results</h3>
            <p className="text-muted-foreground">
              Receive detailed information about the detected raga, including 
              scale patterns, cultural context, and performance guidelines.
            </p>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="text-center space-y-6 py-12 bg-muted/50 rounded-lg">
        <h2 className="text-3xl font-bold">Ready to Discover?</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Start your journey into the world of Indian classical music. 
          Upload your first audio file and let our AI reveal the hidden raga.
        </p>
        <Link href="/detect">
          <Button size="lg" className="text-lg px-8 py-4">
            Start Detecting Ragas
          </Button>
        </Link>
      </section>
    </div>
  )
}
