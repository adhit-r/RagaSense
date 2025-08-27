// API Response Types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// Raga Types
export interface Raga {
  id: number
  name: string
  alternate_names?: string[]
  tradition?: 'Hindustani' | 'Carnatic'
  arohana?: string[]
  avarohana?: string[]
  characteristic_phrases?: string[]
  vadi?: string
  samvadi?: string
  varjya_swaras?: string[]
  jati?: string
  time?: string[]
  season?: string[]
  rasa?: string[]
  mood?: string[]
  description?: string
  history?: string
  notable_compositions?: string[]
  audio_features?: any
  pitch_distribution?: any
  tonic_frequency?: number
  aroha_patterns?: any
  avaroha_patterns?: any
  pakad?: string
  practice_exercises?: string[]
  thaat?: string
  time_period?: string
  regional_style?: any
  melakarta_number?: number
  carnatic_equivalent?: string
  hindustani_equivalent?: string
  janaka_raga?: string
  janya_ragas?: string[]
  chakra?: string
  icon?: string
  melakarta_name?: string
  stats?: any
  info?: any
  songs?: any
  keyboard?: any
}

// Raga Detection Types
export interface RagaPrediction {
  raga: string
  probability: number
  info?: {
    aroha?: string[]
    avaroha?: string[]
    time?: string
    mood?: string
  }
}

export interface RagaDetectionResult {
  predictions: RagaPrediction[]
  audio_features?: any
  processing_time?: number
  confidence_score?: number
}

// Audio Types
export interface AudioFile {
  file: File
  url: string
  name: string
  size: number
  type: string
  duration?: number
}

export interface AudioPlayerState {
  isPlaying: boolean
  currentTime: number
  duration: number
  volume: number
  isMuted: boolean
}

// UI Types
export type Theme = 'light' | 'dark' | 'system'

export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
}

// Form Types
export interface UploadFormData {
  audio: File
  duration?: number
}

// API Error Types
export interface ApiError {
  status: number
  message: string
  details?: any
}

// Pagination Types
export interface PaginationMeta {
  total: number
  pages: number
  current_page: number
  per_page: number
  has_next: boolean
  has_prev: boolean
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: PaginationMeta
}

// Component Props Types
export interface ButtonProps {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link'
  size?: 'default' | 'sm' | 'lg' | 'icon'
  disabled?: boolean
  loading?: boolean
  onClick?: () => void
  children: React.ReactNode
  className?: string
}

export interface CardProps {
  title?: string
  description?: string
  children: React.ReactNode
  className?: string
}

export interface InputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'file'
  placeholder?: string
  value?: string | number
  onChange?: (value: string) => void
  disabled?: boolean
  required?: boolean
  className?: string
}

// Hook Types
export interface UseAudioPlayerReturn {
  audioUrl: string | null
  isPlaying: boolean
  currentTime: number
  duration: number
  play: () => void
  pause: () => void
  reset: () => void
  setAudioUrl: (url: string) => void
  setCurrentTime: (time: number) => void
  setVolume: (volume: number) => void
  toggleMute: () => void
}

export interface UseRagaDetectionReturn {
  detectRaga: (file: File) => Promise<RagaDetectionResult>
  isDetecting: boolean
  error: string | null
  result: RagaDetectionResult | null
}
