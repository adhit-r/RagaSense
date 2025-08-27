import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useRagas() {
  const getAllRagas = useQuery(api.ragas.getAll);
  const searchRagas = useMutation(api.ragas.search);
  const getByTradition = useMutation(api.ragas.getByTradition);
  const getByMood = useMutation(api.ragas.getByMood);
  const getByTime = useMutation(api.ragas.getByTime);
  const getSuggestedByMood = useMutation(api.ragas.getSuggestedByMood);
  const getSuggestedByTheme = useMutation(api.ragas.getSuggestedByTheme);

  return {
    getAllRagas,
    searchRagas,
    getByTradition,
    getByMood,
    getByTime,
    getSuggestedByMood,
    getSuggestedByTheme,
  };
}

export function useRagaByName(name: string) {
  return useQuery(api.ragas.getByName, { name });
}

export function useRagasByTradition(tradition: "Hindustani" | "Carnatic") {
  return useQuery(api.ragas.getByTradition, { tradition });
}
