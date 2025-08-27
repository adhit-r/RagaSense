import { Search, Upload, Music } from 'lucide-react';

export function RagaDetector() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold">Raga Detection</h1>
        <p className="text-xl text-muted-foreground">
          Upload audio files and instantly identify the raga using advanced AI
        </p>
      </div>

      <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
        <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">Upload Audio File</h3>
        <p className="text-muted-foreground mb-4">
          Drag and drop your audio file here, or click to browse
        </p>
        <button className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors">
          Choose File
        </button>
        <p className="text-sm text-muted-foreground mt-4">
          Supports WAV, MP3, OGG, FLAC, M4A (max 50MB)
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="border rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Search className="h-6 w-6 text-green-500" />
            <h3 className="text-lg font-semibold">How It Works</h3>
          </div>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• Upload your audio file</li>
            <li>• AI analyzes the musical patterns</li>
            <li>• Get instant raga identification</li>
            <li>• View detailed scale information</li>
          </ul>
        </div>

        <div className="border rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Music className="h-6 w-6 text-blue-500" />
            <h3 className="text-lg font-semibold">Supported Formats</h3>
          </div>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• WAV (uncompressed)</li>
            <li>• MP3 (compressed)</li>
            <li>• OGG (open format)</li>
            <li>• FLAC (lossless)</li>
            <li>• M4A (AAC)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
