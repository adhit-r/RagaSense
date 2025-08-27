import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useAnalytics() {
  const getUserAnalytics = useQuery(api.analytics.getUserAnalytics);
  const trackEvent = useMutation(api.analytics.trackEvent);
  const trackRagaDetection = useMutation(api.analytics.trackRagaDetection);
  const trackMusicGeneration = useMutation(api.analytics.trackMusicGeneration);
  const trackUserAction = useMutation(api.analytics.trackUserAction);
  const trackError = useMutation(api.analytics.trackError);

  return {
    getUserAnalytics,
    trackEvent,
    trackRagaDetection,
    trackMusicGeneration,
    trackUserAction,
    trackError,
  };
}

export function useEventAnalytics(eventType: string, startDate?: number, endDate?: number) {
  return useQuery(api.analytics.getEventAnalytics, { eventType, startDate, endDate });
}

export function useGlobalAnalytics(startDate?: number, endDate?: number) {
  return useQuery(api.analytics.getGlobalAnalytics, { startDate, endDate });
}
