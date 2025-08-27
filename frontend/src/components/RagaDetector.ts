interface RagaPrediction {
  raga: string;
  probability: number;
  confidence: string;
}

interface DetectionResult {
  success: boolean;
  predicted_raga?: string;
  confidence?: number;
  top_predictions?: RagaPrediction[];
  supported_ragas?: string[];
  error?: string;
}

export class RagaDetector {
  private isUploading = false;
  private result: DetectionResult | null = null;
  private dragActive = false;
  private isListening = false;
  private fileInput: HTMLInputElement | null = null;
  private container: HTMLElement | null = null;

  constructor(containerId: string) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container with id '${containerId}' not found`);
    }
    this.init();
  }

  private init() {
    this.render();
    this.attachEventListeners();
  }

  private async handleFileUpload(file: File) {
    if (!file) return;

    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/m4a'];
    if (!allowedTypes.includes(file.type)) {
      this.result = {
        success: false,
        error: 'Unsupported file type. Please upload WAV, MP3, OGG, FLAC, or M4A files.'
      };
      this.render();
      return;
    }

    this.isUploading = true;
    this.result = null;
    this.render();

    try {
      const formData = new FormData();
      formData.append('audio', file);

      const response = await fetch('/api/ragas/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        this.result = data;
      } else {
        this.result = {
          success: false,
          error: data.detail || 'Failed to detect raga'
        };
      }
    } catch (error) {
      this.result = {
        success: false,
        error: 'Network error. Please try again.'
      };
    } finally {
      this.isUploading = false;
      this.render();
    }
  }

  private handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      this.dragActive = true;
    } else if (e.type === 'dragleave') {
      this.dragActive = false;
    }
    this.render();
  };

  private handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    this.dragActive = false;

    if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
      this.handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  private handleFileSelect = (e: Event) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      this.handleFileUpload(target.files[0]);
    }
  };

  private handleClick = () => {
    this.fileInput?.click();
  };

  private startListening = () => {
    this.isListening = true;
    this.render();
    
    // TODO: Implement microphone recording
    setTimeout(() => {
      this.isListening = false;
      // Simulate detection result
      this.result = {
        success: true,
        predicted_raga: 'Yaman',
        confidence: 0.85,
        top_predictions: [
          { raga: 'Yaman', probability: 0.85, confidence: 'High' },
          { raga: 'Bhairav', probability: 0.10, confidence: 'Low' },
          { raga: 'Kafi', probability: 0.05, confidence: 'Low' }
        ],
        supported_ragas: ['Yaman', 'Bhairav', 'Kafi']
      };
      this.render();
    }, 3000);
  };

  private resetDetection = () => {
    this.result = null;
    this.isUploading = false;
    this.isListening = false;
    this.render();
  };

  private attachEventListeners() {
    // File input change
    this.fileInput = this.container?.querySelector('#file-input') as HTMLInputElement;
    this.fileInput?.addEventListener('change', this.handleFileSelect);

    // Upload area events
    const uploadArea = this.container?.querySelector('#upload-area');
    if (uploadArea) {
      uploadArea.addEventListener('dragenter', this.handleDrag);
      uploadArea.addEventListener('dragleave', this.handleDrag);
      uploadArea.addEventListener('dragover', this.handleDrag);
      uploadArea.addEventListener('drop', this.handleDrop);
      uploadArea.addEventListener('click', this.handleClick);
    }

    // Record button
    const recordButton = this.container?.querySelector('#record-button');
    recordButton?.addEventListener('click', this.startListening);

    // Action buttons
    const tryAgainButton = this.container?.querySelector('#try-again-button');
    tryAgainButton?.addEventListener('click', this.resetDetection);

    const learnMoreButton = this.container?.querySelector('#learn-more-button');
    learnMoreButton?.addEventListener('click', () => {
      window.open('https://en.wikipedia.org/wiki/Raga', '_blank');
    });
  }

  private render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100">
        <div class="max-w-4xl mx-auto px-4 py-8">
          <!-- Header -->
          <div class="text-center mb-12">
            <div class="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-purple-500 to-blue-600 rounded-full mb-6">
              <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
            </div>
            <h1 class="text-4xl font-bold text-gray-900 mb-4">
              Raga Recognition
            </h1>
            <p class="text-xl text-gray-600 max-w-2xl mx-auto">
              Discover the raga in Indian classical music. Upload an audio file or record live to identify the raga instantly.
            </p>
          </div>

          <!-- Main Content -->
          <div class="bg-white rounded-3xl shadow-xl p-8 mb-8">
            ${!this.result ? this.renderUploadInterface() : this.renderResultsInterface()}
          </div>

          <!-- Footer Info -->
          <div class="text-center text-gray-500 text-sm">
            <p>
              Powered by advanced machine learning • Currently supporting 3 ragas • 
              <a href="#" class="text-purple-600 hover:text-purple-700 ml-1">Learn about our technology</a>
            </p>
          </div>
        </div>
      </div>
    `;

    // Re-attach event listeners after re-render
    this.attachEventListeners();
  }

  private renderUploadInterface(): string {
    return `
      <div class="space-y-8">
        <!-- Upload Area -->
        <div
          id="upload-area"
          class="relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
            this.dragActive
              ? 'border-purple-500 bg-purple-50 scale-105'
              : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
          } ${this.isUploading ? 'pointer-events-none opacity-75' : 'cursor-pointer'}"
        >
          <input
            id="file-input"
            type="file"
            accept="audio/*"
            class="hidden"
          />
          
          <div class="space-y-6">
            <div class="relative">
              <div class="w-24 h-24 mx-auto rounded-full flex items-center justify-center transition-all duration-300 ${
                this.isUploading 
                  ? 'bg-purple-100 animate-pulse' 
                  : 'bg-gradient-to-r from-purple-100 to-blue-100'
              }">
                ${this.isUploading ? `
                  <div class="w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                ` : `
                  <svg class="w-12 h-12 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                `}
              </div>
            </div>
            
            <div>
              <h3 class="text-2xl font-semibold text-gray-900 mb-2">
                ${this.isUploading ? 'Analyzing...' : 'Upload Audio File'}
              </h3>
              <p class="text-gray-600 mb-4">
                ${this.isUploading 
                  ? 'Processing your audio file to detect the raga'
                  : 'Drag and drop your audio file here, or click to browse'
                }
              </p>
              ${!this.isUploading ? `
                <p class="text-sm text-gray-500">
                  Supports WAV, MP3, OGG, FLAC, M4A • Max 30 seconds
                </p>
              ` : ''}
            </div>
          </div>
        </div>

        <!-- Divider -->
        <div class="relative">
          <div class="absolute inset-0 flex items-center">
            <div class="w-full border-t border-gray-200"></div>
          </div>
          <div class="relative flex justify-center text-sm">
            <span class="px-4 bg-white text-gray-500">or</span>
          </div>
        </div>

        <!-- Record Button -->
        <div class="text-center">
          <button
            id="record-button"
            disabled="${this.isListening}"
            class="inline-flex items-center px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 ${
              this.isListening
                ? 'bg-red-500 text-white animate-pulse'
                : 'bg-gradient-to-r from-purple-500 to-blue-600 text-white hover:from-purple-600 hover:to-blue-700 transform hover:scale-105'
            }"
          >
            ${this.isListening ? `
              <div class="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin mr-3"></div>
              Listening...
            ` : `
              <svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              Record Live
            `}
          </button>
        </div>
      </div>
    `;
  }

  private renderResultsInterface(): string {
    if (!this.result) return '';

    if (this.result.success) {
      return `
        <div class="space-y-8">
          <!-- Success Result -->
          <div class="text-center">
            <div class="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-green-500 to-emerald-600 rounded-full mb-6">
              <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
            </div>
            
            <h2 class="text-3xl font-bold text-gray-900 mb-2">
              Raga Detected!
            </h2>
            
            <div class="bg-gradient-to-r from-purple-500 to-blue-600 text-white px-8 py-4 rounded-2xl inline-block mb-6">
              <div class="text-4xl font-bold mb-1">
                ${this.result.predicted_raga}
              </div>
              <div class="text-lg opacity-90">
                Confidence: ${(this.result.confidence! * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          ${this.renderTopPredictions()}
          ${this.renderSupportedRagas()}
          ${this.renderActionButtons()}
        </div>
      `;
    } else {
      return `
        <div class="space-y-8">
          <!-- Error Result -->
          <div class="text-center">
            <div class="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-red-500 to-pink-600 rounded-full mb-6">
              <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            
            <h2 class="text-3xl font-bold text-gray-900 mb-4">
              Detection Failed
            </h2>
            
            <div class="bg-red-50 border border-red-200 rounded-2xl p-6 max-w-md mx-auto">
              <p class="text-red-800">${this.result.error}</p>
            </div>
          </div>
          ${this.renderActionButtons()}
        </div>
      `;
    }
  }

  private renderTopPredictions(): string {
    if (!this.result?.top_predictions?.length) return '';

    return `
      <div>
        <h3 class="text-xl font-semibold text-gray-900 mb-4">
          All Predictions
        </h3>
        <div class="space-y-3">
          ${this.result.top_predictions.map((prediction, index) => `
            <div class="flex items-center justify-between p-4 rounded-xl border-2 transition-all duration-300 ${
              index === 0
                ? 'border-purple-200 bg-purple-50'
                : 'border-gray-200 bg-gray-50'
            }">
              <div class="flex items-center space-x-4">
                <div class="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                  index === 0
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-300 text-gray-700'
                }">
                  ${index + 1}
                </div>
                <div>
                  <div class="font-semibold text-gray-900">
                    ${prediction.raga}
                  </div>
                  <div class="text-sm text-gray-600">
                    ${prediction.confidence} confidence
                  </div>
                </div>
              </div>
              <div class="text-right">
                <div class="text-lg font-bold text-gray-900">
                  ${(prediction.probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  private renderSupportedRagas(): string {
    if (!this.result?.supported_ragas?.length) return '';

    return `
      <div>
        <h3 class="text-xl font-semibold text-gray-900 mb-4">
          Supported Ragas
        </h3>
        <div class="flex flex-wrap gap-3">
          ${this.result.supported_ragas.map(raga => `
            <span class="px-4 py-2 bg-gradient-to-r from-purple-100 to-blue-100 text-purple-800 rounded-full text-sm font-medium">
              ${raga}
            </span>
          `).join('')}
        </div>
      </div>
    `;
  }

  private renderActionButtons(): string {
    return `
      <div class="text-center space-x-4">
        <button
          id="try-again-button"
          class="inline-flex items-center px-6 py-3 bg-gray-100 text-gray-700 rounded-xl font-semibold hover:bg-gray-200 transition-colors duration-200"
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Try Again
        </button>
        
        <button
          id="learn-more-button"
          class="inline-flex items-center px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-600 text-white rounded-xl font-semibold hover:from-purple-600 hover:to-blue-700 transition-all duration-200"
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
          Learn More
        </button>
      </div>
    `;
  }
}
