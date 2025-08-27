import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Mic, Upload, BookOpen } from 'lucide-react';

export function Home() {
  return (
    <div className="space-y-12">
      <section className="text-center space-y-6 py-20">
        <h1 className="text-5xl font-bold tracking-tight sm:text-6xl bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
          Discover the Magic of Indian Classical Music
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          Upload an audio recording and let our AI identify the raga, explore its characteristics,
          and learn about its rich musical heritage.
        </p>
        <div className="flex items-center justify-center gap-4 pt-4">
          <Button size="lg" asChild>
            <Link to="/detect">
              <Upload className="mr-2 h-4 w-4" />
              Detect Raga
            </Link>
          </Button>
          <Button variant="outline" size="lg" asChild>
            <Link to="/ragas">
              <BookOpen className="mr-2 h-4 w-4" />
              Explore Ragas
            </Link>
          </Button>
        </div>
      </section>

      <section className="py-12">
        <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: <Upload className="h-8 w-8 text-primary" />,
              title: 'Upload Audio',
              description: 'Upload a recording or record directly from your microphone.'
            },
            {
              icon: <Mic className="h-8 w-8 text-primary" />,
              title: 'Analyze',
              description: 'Our AI analyzes the audio to identify the raga and its characteristics.'
            },
            {
              icon: <BookOpen className="h-8 w-8 text-primary" />,
              title: 'Learn',
              description: 'Explore detailed information about the raga, including its scale, time, and mood.'
            }
          ].map((step, index) => (
            <div key={index} className="flex flex-col items-center text-center p-6 rounded-lg bg-card border">
              <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary/10 mb-4">
                {step.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
              <p className="text-muted-foreground">{step.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="py-12 bg-muted/50 rounded-lg">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-8">Popular Ragas</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { name: 'Yaman', time: 'Evening', mood: 'Peaceful, Romantic' },
              { name: 'Bhairav', time: 'Morning', mood: 'Devotional, Serious' },
              { name: 'Malkauns', time: 'Night', mood: 'Serious, Mystical' },
              { name: 'Kafi', time: 'Late Evening', mood: 'Light, Romantic' },
              { name: 'Todi', time: 'Morning', mood: 'Serious, Sober' },
              { name: 'Bhairavi', time: 'Morning', mood: 'Devotional, Peaceful' },
            ].map((raga, index) => (
              <div key={index} className="p-6 bg-background rounded-lg border">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-xl font-semibold">{raga.name}</h3>
                    <p className="text-muted-foreground">{raga.time}</p>
                  </div>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                    {raga.mood}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="text-center mt-8">
            <Button variant="outline" asChild>
              <Link to="/ragas">View All Ragas</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
