import { query } from "./_generated/server";
import { v } from "convex/values";

// Get multi-model detection history
export const getMultiModelDetectionHistory = query({
  args: {
    userId: v.string(),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const detections = await ctx.db
      .query("multiModelDetections")
      .withIndex("by_user", (q) => q.eq("userId", args.userId))
      .order("desc")
      .take(args.limit || 50);

    return detections.map(detection => ({
      audioFileId: detection.audioFileId,
      results: detection.results,
      processingTime: detection.processingTime,
      timestamp: detection.timestamp,
    }));
  },
});

// Get single multi-model detection
export const getMultiModelDetection = query({
  args: {
    detectionId: v.id("multiModelDetections"),
  },
  handler: async (ctx, args) => {
    const detection = await ctx.db.get(args.detectionId);
    
    if (!detection) {
      throw new Error("Detection not found");
    }

    return {
      audioFileId: detection.audioFileId,
      results: detection.results,
      processingTime: detection.processingTime,
      timestamp: detection.timestamp,
    };
  },
});

// Get detection statistics for user
export const getDetectionStats = query({
  args: {
    userId: v.string(),
  },
  handler: async (ctx, args) => {
    const detections = await ctx.db
      .query("multiModelDetections")
      .withIndex("by_user", (q) => q.eq("userId", args.userId))
      .collect();

    const stats = {
      totalDetections: detections.length,
      totalModels: 0,
      averageProcessingTime: 0,
      mostCommonRagas: {} as Record<string, number>,
      modelUsage: {} as Record<string, number>,
    };

    if (detections.length === 0) {
      return stats;
    }

    let totalProcessingTime = 0;
    const ragaCounts: Record<string, number> = {};
    const modelCounts: Record<string, number> = {};

    for (const detection of detections) {
      // Parse processing time (assuming format like "150ms" or "1.5s")
      const timeStr = detection.processingTime;
      let timeMs = 0;
      if (timeStr.includes("ms")) {
        timeMs = parseFloat(timeStr.replace("ms", ""));
      } else if (timeStr.includes("s")) {
        timeMs = parseFloat(timeStr.replace("s", "")) * 1000;
      }
      totalProcessingTime += timeMs;

      // Count ragas and models
      for (const result of detection.results) {
        if (!result.error) {
          ragaCounts[result.raga] = (ragaCounts[result.raga] || 0) + 1;
          modelCounts[result.modelName] = (modelCounts[result.modelName] || 0) + 1;
        }
      }
    }

    stats.totalModels = Object.keys(modelCounts).length;
    stats.averageProcessingTime = totalProcessingTime / detections.length;
    stats.mostCommonRagas = ragaCounts;
    stats.modelUsage = modelCounts;

    return stats;
  },
});
