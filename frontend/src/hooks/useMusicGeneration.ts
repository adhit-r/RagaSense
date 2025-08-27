import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useMusicGeneration() {
  const getMusicGenerationHistory = useQuery(api.musicGeneration.getMusicGenerationHistory, { limit: 20 });
  const getMusicGenerationStats = useQuery(api.musicGeneration.getMusicGenerationStats);
  const createMusicGeneration = useMutation(api.musicGeneration.createMusicGeneration);
  const updateMusicGenerationStatus = useMutation(api.musicGeneration.updateMusicGenerationStatus);
  const deleteMusicGeneration = useMutation(api.musicGeneration.deleteMusicGeneration);
  const exportMusicGenerationHistory = useMutation(api.musicGeneration.exportMusicGenerationHistory);

  return {
    getMusicGenerationHistory,
    getMusicGenerationStats,
    createMusicGeneration,
    updateMusicGenerationStatus,
    deleteMusicGeneration,
    exportMusicGenerationHistory,
  };
}

export function useMusicGenerationById(generationId: string) {
  return useQuery(api.musicGeneration.getMusicGenerationById, { generationId });
}

export function useMusicGenerationHistoryPaginated(limit: number = 20, offset: number = 0) {
  return useQuery(api.musicGeneration.getMusicGenerationHistory, { limit, offset });
}

export function usePendingGenerations() {
  return useQuery(api.musicGeneration.getPendingGenerations);
}
