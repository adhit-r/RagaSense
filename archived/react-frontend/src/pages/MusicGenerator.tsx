import { Sparkles, Music, Mic, Settings } from 'lucide-react';

export function MusicGenerator() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold">AI Music Generator</h1>
        <p className="text-xl text-muted-foreground">
          Create personalized Indian classical music using AI
        </p>
      </div>

      {/* Step Indicator */}
      <div className="flex justify-center">
        <div className="flex space-x-4">
          {[1, 2, 3, 4, 5].map((step) => (
            <div
              key={step}
              className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                step === 1
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground'
              }`}
            >
              {step}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: Music Type Selection */}
      <div className="border rounded-lg p-8">
        <h2 className="text-2xl font-bold mb-6">Step 1: Choose Music Type</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <button className="border-2 border-gray-300 rounded-lg p-8 text-center hover:border-primary hover:bg-primary/5 transition-all">
            <Music className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Instrumental</h3>
            <p className="text-muted-foreground">
              Choose from classical instruments
            </p>
          </button>
          
          <button className="border-2 border-gray-300 rounded-lg p-8 text-center hover:border-primary hover:bg-primary/5 transition-all">
            <Mic className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Vocal</h3>
            <p className="text-muted-foreground">
              Select voice characteristics
            </p>
          </button>
        </div>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="border rounded-lg p-6 text-center">
          <Sparkles className="h-8 w-8 text-purple-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">AI-Powered</h3>
          <p className="text-sm text-muted-foreground">
            Advanced AI algorithms create authentic Indian classical music
          </p>
        </div>

        <div className="border rounded-lg p-6 text-center">
          <Settings className="h-8 w-8 text-blue-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Customizable</h3>
          <p className="text-sm text-muted-foreground">
            Choose instruments, voices, moods, and themes
          </p>
        </div>

        <div className="border rounded-lg p-6 text-center">
          <Music className="h-8 w-8 text-green-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Authentic</h3>
          <p className="text-sm text-muted-foreground">
            Based on traditional raga structures and musical theory
          </p>
        </div>
      </div>

      {/* How It Works */}
      <div className="bg-muted/50 rounded-lg p-8">
        <h2 className="text-2xl font-bold mb-6 text-center">How It Works</h2>
        <div className="grid md:grid-cols-5 gap-4">
          {[
            { step: '1', title: 'Choose Type', desc: 'Instrumental or Vocal' },
            { step: '2', title: 'Select Options', desc: 'Instruments or Voice' },
            { step: '3', title: 'Pick Mood', desc: 'Emotional character' },
            { step: '4', title: 'Choose Theme', desc: 'Context and style' },
            { step: '5', title: 'Generate', desc: 'AI creates your music' },
          ].map((item) => (
            <div key={item.step} className="text-center">
              <div className="w-12 h-12 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-bold mx-auto mb-3">
                {item.step}
              </div>
              <h3 className="font-semibold mb-1">{item.title}</h3>
              <p className="text-sm text-muted-foreground">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
