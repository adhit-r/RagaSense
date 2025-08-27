import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useRagas() {
  const getAllRagas = useQuery(api.ragas.getAllRagas);
  const getUserFavorites = useQuery(api.ragas.getUserFavorites);
  const addToFavorites = useMutation(api.ragas.addToFavorites);
  const removeFromFavorites = useMutation(api.ragas.removeFromFavorites);
  const searchRagas = useMutation(api.ragas.searchRagas);

  return {
    getAllRagas,
    getUserFavorites,
    addToFavorites,
    removeFromFavorites,
    searchRagas,
  };
}

export function useRagaByName(name: string) {
  return useQuery(api.ragas.getRagaByName, { name });
}

export function useRagasByCategory(category: string) {
  return useQuery(api.ragas.getRagasByCategory, { category });
}

export function useRagasByTimeOfDay(timeOfDay: "morning" | "afternoon" | "evening" | "night") {
  return useQuery(api.ragas.getRagasByTimeOfDay, { timeOfDay });
}

export function useIsFavorited(ragaId: string) {
  return useQuery(api.ragas.isFavorited, { ragaId });
}
