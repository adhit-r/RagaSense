import { apiClient } from './client'
import { 
  MusicGenerationRequest, 
  MusicGenerationResponse, 
  MusicGenerationHistory,
  MoodCategory,
  ThemeCategory
} from '@/types/music-generation'

// Generate music
export async function generateMusic(
  request: MusicGenerationRequest
): Promise<MusicGenerationResponse> {
  const response = await apiClient.post<MusicGenerationResponse>(
    '/music/generate',
    request
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to generate music')
  }
  
  return response.data!
}

// Get generation status
export async function getGenerationStatus(
  generationId: string
): Promise<MusicGenerationResponse> {
  const response = await apiClient.get<MusicGenerationResponse>(
    `/music/generate/${generationId}/status`
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to get generation status')
  }
  
  return response.data!
}

// Get generation history
export async function getGenerationHistory(
  params?: {
    page?: number
    per_page?: number
    isFavorite?: boolean
  }
): Promise<{
  data: MusicGenerationHistory[]
  pagination: {
    total: number
    pages: number
    current_page: number
    per_page: number
    has_next: boolean
    has_prev: boolean
  }
}> {
  const response = await apiClient.get<{
    data: MusicGenerationHistory[]
    pagination: {
      total: number
      pages: number
      current_page: number
      per_page: number
      has_next: boolean
      has_prev: boolean
    }
  }>('/music/history', params)
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to get generation history')
  }
  
  return response.data!
}

// Toggle favorite status
export async function toggleFavorite(
  generationId: string
): Promise<{ isFavorite: boolean }> {
  const response = await apiClient.put<{ isFavorite: boolean }>(
    `/music/history/${generationId}/favorite`
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to toggle favorite')
  }
  
  return response.data!
}

// Delete generation
export async function deleteGeneration(
  generationId: string
): Promise<void> {
  const response = await apiClient.delete<void>(
    `/music/history/${generationId}`
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to delete generation')
  }
}

// Get suggested ragas by mood
export async function getSuggestedRagasByMood(
  mood: MoodCategory
): Promise<string[]> {
  const response = await apiClient.get<{ ragas: string[] }>(
    `/music/suggestions/mood/${mood}`
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to get suggested ragas')
  }
  
  return response.data!.ragas
}

// Get suggested ragas by theme
export async function getSuggestedRagasByTheme(
  theme: ThemeCategory
): Promise<string[]> {
  const response = await apiClient.get<{ ragas: string[] }>(
    `/music/suggestions/theme/${theme}`
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to get suggested ragas')
  }
  
  return response.data!.ragas
}

// Get available instruments
export async function getAvailableInstruments(): Promise<{
  classical: string[]
  fusion: string[]
  devotional: string[]
  instrumental: string[]
  vocal: string[]
}> {
  const response = await apiClient.get<{
    classical: string[]
    fusion: string[]
    devotional: string[]
    instrumental: string[]
    vocal: string[]
  }>('/music/instruments')
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to get instruments')
  }
  
  return response.data!
}

// Get voice characteristics
export async function getVoiceCharacteristics(): Promise<{
  male: Record<string, { range: string; style: string }>
  female: Record<string, { range: string; style: string }>
}> {
  const response = await apiClient.get<{
    male: Record<string, { range: string; style: string }>
    female: Record<string, { range: string; style: string }>
  }>('/music/voice-characteristics')
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to get voice characteristics')
  }
  
  return response.data!
}

// Download generated music
export async function downloadMusic(
  generationId: string,
  format: 'mp3' | 'wav' | 'flac' = 'mp3'
): Promise<Blob> {
  const response = await fetch(`${apiClient['baseUrl']}/music/download/${generationId}?format=${format}`)
  
  if (!response.ok) {
    throw new Error('Failed to download music')
  }
  
  return response.blob()
}

// Share generated music
export async function shareMusic(
  generationId: string,
  platform: 'social' | 'email' | 'link'
): Promise<{ shareUrl: string }> {
  const response = await apiClient.post<{ shareUrl: string }>(
    `/music/share/${generationId}`,
    { platform }
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to share music')
  }
  
  return response.data!
}
