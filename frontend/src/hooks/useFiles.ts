import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useFiles() {
  const getUserFiles = useQuery(api.files.getUserFiles, { limit: 20 });
  const getFileStats = useQuery(api.files.getFileStats);
  const createFile = useMutation(api.files.createFile);
  const updateFile = useMutation(api.files.updateFile);
  const deleteFile = useMutation(api.files.deleteFile);
  const searchFiles = useMutation(api.files.searchFiles);

  return {
    getUserFiles,
    getFileStats,
    createFile,
    updateFile,
    deleteFile,
    searchFiles,
  };
}

export function useFileById(fileId: string) {
  return useQuery(api.files.getFileById, { fileId });
}

export function useFilesByType(type: string) {
  return useQuery(api.files.getFilesByType, { type });
}

export function useUserFilesPaginated(limit: number = 20, offset: number = 0) {
  return useQuery(api.files.getUserFiles, { limit, offset });
}
