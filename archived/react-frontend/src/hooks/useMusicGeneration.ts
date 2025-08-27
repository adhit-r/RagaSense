import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useMusicGeneration() {
  const startGeneration = useMutation(api.musicGeneration.startGeneration);
  const getUserHistory = useQuery(api.musicGeneration.getUserHistory);
  const toggleFavorite = useMutation(api.musicGeneration.toggleFavorite);
  const deleteGeneration = useMutation(api.musicGeneration.deleteGeneration);
  const getStats = useQuery(api.musicGeneration.getStats);

  return {
    startGeneration,
    getUserHistory,
    toggleFavorite,
    deleteGeneration,
    getStats,
  };
}

export function useGeneration(generationId: string) {
  return useQuery(api.musicGeneration.getGeneration, { generationId });
}

export function useUserHistory(isFavorite?: boolean) {
  return useQuery(api.musicGeneration.getUserHistory, { isFavorite });
}
