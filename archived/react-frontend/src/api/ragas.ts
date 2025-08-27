import { apiClient } from './client'
import { Raga, RagaDetectionResult, PaginatedResponse } from '@/types'

// Raga Detection
export async function detectRaga(
  file: File,
  onProgress?: (progress: number) => void
): Promise<RagaDetectionResult> {
  const response = await apiClient.upload<RagaDetectionResult>(
    '/ragas/detect',
    file,
    onProgress
  )
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to detect raga')
  }
  
  return response.data!
}

// Get all ragas with pagination
export async function getRagas(params?: {
  page?: number
  per_page?: number
  search?: string
  tradition?: 'Hindustani' | 'Carnatic'
}): Promise<PaginatedResponse<Raga>> {
  const response = await apiClient.get<PaginatedResponse<Raga>>('/ragas', params)
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch ragas')
  }
  
  return response.data!
}

// Get a single raga by ID
export async function getRaga(id: number): Promise<Raga> {
  const response = await apiClient.get<Raga>(`/ragas/${id}`)
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch raga')
  }
  
  return response.data!
}

// Get supported ragas (for detection model)
export async function getSupportedRagas(): Promise<string[]> {
  const response = await apiClient.get<{ ragas: string[] }>('/ragas/supported-ragas')
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch supported ragas')
  }
  
  return response.data!.ragas
}

// Search ragas
export async function searchRagas(query: string): Promise<Raga[]> {
  const response = await apiClient.get<Raga[]>('/ragas/search', { q: query })
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to search ragas')
  }
  
  return response.data!
}

// Get raga analysis
export async function getRagaAnalysis(ragaId: number): Promise<any> {
  const response = await apiClient.get<any>(`/ragas/${ragaId}/analysis`)
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch raga analysis')
  }
  
  return response.data!
}

// Compare ragas
export async function compareRagas(raga1Id: number, raga2Id: number): Promise<any> {
  const response = await apiClient.get<any>('/ragas/compare', {
    raga1_id: raga1Id,
    raga2_id: raga2Id
  })
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to compare ragas')
  }
  
  return response.data!
}

// Get ragas by tradition
export async function getRagasByTradition(tradition: 'Hindustani' | 'Carnatic'): Promise<Raga[]> {
  const response = await apiClient.get<Raga[]>('/ragas', { tradition })
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch ragas by tradition')
  }
  
  return response.data!
}

// Get ragas by time of day
export async function getRagasByTime(time: string): Promise<Raga[]> {
  const response = await apiClient.get<Raga[]>('/ragas', { time })
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch ragas by time')
  }
  
  return response.data!
}

// Get ragas by mood
export async function getRagasByMood(mood: string): Promise<Raga[]> {
  const response = await apiClient.get<Raga[]>('/ragas', { mood })
  
  if (!response.success) {
    throw new Error(response.message || 'Failed to fetch ragas by mood')
  }
  
  return response.data!
}
