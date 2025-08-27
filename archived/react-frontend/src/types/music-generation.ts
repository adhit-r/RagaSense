// Music Generation Types

export type MusicType = 'instrumental' | 'vocal'

export type InstrumentType = 
  | 'sitar' | 'tabla' | 'flute' | 'veena' | 'santoor' 
  | 'harmonium' | 'violin' | 'guitar' | 'tanpura' | 'sarangi'

export type VoiceGender = 'male' | 'female'

export type VoicePitch = 'high' | 'medium' | 'low'

export type VoiceStyle = 'classical' | 'semi-classical' | 'devotional'

export type MoodCategory = 
  | 'peaceful' | 'joyful' | 'romantic' | 'energetic' | 'melancholic'

export type ThemeCategory = 
  | 'spiritual' | 'cultural' | 'contemporary' | 'educational'

export type SpiritualTheme = 'morning-prayer' | 'evening-devotion' | 'meditation'
export type CulturalTheme = 'festival' | 'seasonal' | 'regional'
export type ContemporaryTheme = 'fusion' | 'ambient' | 'dance'
export type EducationalTheme = 'learning' | 'practice' | 'improvisation'

export interface VoiceSelection {
  gender: VoiceGender
  pitch: VoicePitch
  style: VoiceStyle
}

export interface InstrumentSelection {
  primary: InstrumentType
  secondary?: InstrumentType[]
  ensemble?: boolean
}

export interface MoodSelection {
  category: MoodCategory
  intensity: number // 1-10
  suggestedRagas: string[]
  selectedRaga?: string
}

export interface ThemeSelection {
  category: ThemeCategory
  subcategory?: SpiritualTheme | CulturalTheme | ContemporaryTheme | EducationalTheme
  context?: string
}

export interface MusicGenerationRequest {
  musicType: MusicType
  instruments?: InstrumentSelection
  voice?: VoiceSelection
  mood: MoodSelection
  theme: ThemeSelection
  duration: number // in seconds
  tempo?: number // BPM
  key?: string
}

export interface MusicGenerationResponse {
  id: string
  status: 'processing' | 'completed' | 'failed'
  audioUrl?: string
  metadata: {
    raga: string
    instruments: string[]
    duration: number
    tempo: number
    key: string
    mood: string
    theme: string
  }
  progress?: number // 0-100
  error?: string
  createdAt: string
  completedAt?: string
}

export interface MusicGenerationHistory {
  id: string
  request: MusicGenerationRequest
  response: MusicGenerationResponse
  createdAt: string
  isFavorite: boolean
}

// Mood-Raga Mapping
export const MOOD_RAGA_MAPPING: Record<MoodCategory, string[]> = {
  peaceful: ['Yaman', 'Bhairav', 'Malkauns', 'Bageshri', 'Darbari'],
  joyful: ['Bilawal', 'Kafi', 'Bhairavi', 'Des', 'Khamaj'],
  romantic: ['Khamaj', 'Des', 'Bageshri', 'Pilu', 'Tilak Kamod'],
  energetic: ['Jog', 'Hansdhwani', 'Shivaranjani', 'Durga', 'Miyan Malhar'],
  melancholic: ['Darbari', 'Marwa', 'Puriya', 'Malkauns', 'Bhairav']
}

// Theme-Raga Mapping
export const THEME_RAGA_MAPPING: Record<ThemeCategory, string[]> = {
  spiritual: ['Bhairav', 'Yaman', 'Malkauns', 'Bageshri', 'Darbari'],
  cultural: ['Bilawal', 'Des', 'Kafi', 'Bhairavi', 'Khamaj'],
  contemporary: ['Khamaj', 'Des', 'Pilu', 'Tilak Kamod', 'Hansdhwani'],
  educational: ['Bilawal', 'Yaman', 'Kafi', 'Bhairav', 'Khamaj']
}

// Instrument Ensembles
export const INSTRUMENT_ENSEMBLES = {
  classical: ['sitar', 'tabla', 'tanpura'],
  fusion: ['sitar', 'guitar', 'tabla'],
  devotional: ['harmonium', 'tabla', 'tanpura'],
  instrumental: ['flute', 'tabla', 'tanpura'],
  vocal: ['harmonium', 'tabla', 'tanpura']
} as const

// Voice Characteristics
export const VOICE_CHARACTERISTICS = {
  male: {
    high: { range: 'C3-C5', style: 'Tenor' },
    medium: { range: 'A2-A4', style: 'Baritone' },
    low: { range: 'F2-F4', style: 'Bass' }
  },
  female: {
    high: { range: 'C4-C6', style: 'Soprano' },
    medium: { range: 'A3-A5', style: 'Alto' },
    low: { range: 'F3-F5', style: 'Mezzo' }
  }
} as const
