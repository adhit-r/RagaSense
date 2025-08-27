import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useAuth as useConvexAuth } from "convex/react";

export function useAuth() {
  const { isAuthenticated, isLoading } = useConvexAuth();
  
  const user = useQuery(api.auth.getCurrentUser);
  const createOrUpdateUser = useMutation(api.auth.createOrUpdateUser);
  const getUserSettings = useQuery(api.auth.getUserSettings);
  const updateUserSettings = useMutation(api.auth.updateUserSettings);
  const deleteUser = useMutation(api.auth.deleteUser);

  return {
    isAuthenticated,
    isLoading,
    user,
    createOrUpdateUser,
    getUserSettings,
    updateUserSettings,
    deleteUser,
  };
}
