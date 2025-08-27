import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useRagaDetection() {
  const getDetectionHistory = useQuery(api.ragaDetection.getDetectionHistory, { limit: 20 });
  const getDetectionStats = useQuery(api.ragaDetection.getDetectionStats);
  const saveDetectionResult = useMutation(api.ragaDetection.saveDetectionResult);
  const deleteDetectionHistory = useMutation(api.ragaDetection.deleteDetectionHistory);
  const exportDetectionHistory = useMutation(api.ragaDetection.exportDetectionHistory);

  return {
    getDetectionHistory,
    getDetectionStats,
    saveDetectionResult,
    deleteDetectionHistory,
    exportDetectionHistory,
  };
}

export function useDetectionById(detectionId: string) {
  return useQuery(api.ragaDetection.getDetectionById, { detectionId });
}

export function useDetectionHistoryPaginated(limit: number = 20, offset: number = 0) {
  return useQuery(api.ragaDetection.getDetectionHistory, { limit, offset });
}
