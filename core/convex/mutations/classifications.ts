import { mutation } from "./_generated/server";
import { v } from "convex/values";

// Create multi-model detection record
export const createMultiModelDetection = mutation({
  args: {
    userId: v.string(),
    audioFileId: v.string(),
    results: v.array(v.object({
      modelName: v.string(),
      raga: v.string(),
      confidence: v.number(),
      tonic: v.optional(v.string()),
      tradition: v.optional(v.string()),
      topPredictions: v.optional(v.array(v.object({
        raga: v.string(),
        confidence: v.number(),
        tradition: v.optional(v.string()),
      }))),
      metadata: v.optional(v.any()),
      error: v.optional(v.string()),
    })),
    processingTime: v.string(),
    timestamp: v.string(),
  },
  handler: async (ctx, args) => {
    const detectionId = await ctx.db.insert("multiModelDetections", {
      userId: args.userId,
      audioFileId: args.audioFileId,
      results: args.results,
      processingTime: args.processingTime,
      timestamp: args.timestamp,
      createdAt: new Date().toISOString(),
    });

    return detectionId;
  },
});

// Get multi-model detection history
export const getMultiModelDetectionHistory = mutation({
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
export const getMultiModelDetection = mutation({
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

// Delete multi-model detection
export const deleteMultiModelDetection = mutation({
  args: {
    detectionId: v.id("multiModelDetections"),
    userId: v.string(),
  },
  handler: async (ctx, args) => {
    const detection = await ctx.db.get(args.detectionId);
    
    if (!detection) {
      throw new Error("Detection not found");
    }

    if (detection.userId !== args.userId) {
      throw new Error("Unauthorized to delete this detection");
    }

    await ctx.db.delete(args.detectionId);
    return { success: true };
  },
});
