import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useFiles() {
  const generateUploadUrl = useMutation(api.files.generateUploadUrl);
  const deleteFile = useMutation(api.files.deleteFile);
  const uploadAudioSample = useMutation(api.files.uploadAudioSample);
  const deleteAudioSample = useMutation(api.files.deleteAudioSample);

  return {
    generateUploadUrl,
    deleteFile,
    uploadAudioSample,
    deleteAudioSample,
  };
}

export function useFileUrl(storageId: string) {
  return useQuery(api.files.getFileUrl, { storageId });
}

export function useUserAudioSamples(isPublic?: boolean) {
  return useQuery(api.files.getUserAudioSamples, { isPublic });
}

export function usePublicAudioSamples(ragaId?: string) {
  return useQuery(api.files.getPublicAudioSamples, { ragaId });
}
