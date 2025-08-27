import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { ConvexError } from "convex/values";

// Get user's raga detection history
export const getDetectionHistory = query({
  args: {
    limit: v.optional(v.number()),
    offset: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      return [];
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      return [];
    }

    const limit = args.limit ?? 20;
    const offset = args.offset ?? 0;

    const detections = await ctx.db
      .query("ragaDetections")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .paginate({
        numItems: limit,
        cursor: null,
      });

    // Get file information for each detection
    const detectionsWithFiles = await Promise.all(
      detections.page.map(async (detection) => {
        const file = await ctx.db.get(detection.audioFileId);
        return {
          ...detection,
          file,
        };
      })
    );

    return detectionsWithFiles;
  },
});

// Save raga detection result
export const saveDetectionResult = mutation({
  args: {
    audioFileId: v.id("files"),
    predictedRaga: v.string(),
    confidence: v.number(),
    topPredictions: v.array(v.object({
      raga: v.string(),
      probability: v.number(),
      confidence: v.string(),
    })),
    processingTime: v.number(),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new ConvexError("Not authenticated");
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      throw new ConvexError("User not found");
    }

    return await ctx.db.insert("ragaDetections", {
      userId: user._id,
      audioFileId: args.audioFileId,
      predictedRaga: args.predictedRaga,
      confidence: args.confidence,
      topPredictions: args.topPredictions,
      processingTime: args.processingTime,
      createdAt: Date.now(),
    });
  },
});

// Get detection statistics
export const getDetectionStats = query({
  args: {},
  handler: async (ctx) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      return null;
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      return null;
    }

    const detections = await ctx.db
      .query("ragaDetections")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();

    // Calculate statistics
    const totalDetections = detections.length;
    const avgConfidence = detections.length > 0 
      ? detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length 
      : 0;
    
    const avgProcessingTime = detections.length > 0
      ? detections.reduce((sum, d) => sum + d.processingTime, 0) / detections.length
      : 0;

    // Count detections by raga
    const ragaCounts = detections.reduce((acc, detection) => {
      acc[detection.predictedRaga] = (acc[detection.predictedRaga] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get most detected ragas
    const topRagas = Object.entries(ragaCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([raga, count]) => ({ raga, count }));

    return {
      totalDetections,
      avgConfidence,
      avgProcessingTime,
      topRagas,
      recentDetections: detections
        .sort((a, b) => b.createdAt - a.createdAt)
        .slice(0, 10),
    };
  },
});

// Delete detection history
export const deleteDetectionHistory = mutation({
  args: {
    detectionId: v.optional(v.id("ragaDetections")),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new ConvexError("Not authenticated");
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      throw new ConvexError("User not found");
    }

    if (args.detectionId) {
      // Delete specific detection
      const detection = await ctx.db.get(args.detectionId);
      if (!detection || detection.userId !== user._id) {
        throw new ConvexError("Detection not found or access denied");
      }
      await ctx.db.delete(args.detectionId);
    } else {
      // Delete all user's detections
      const detections = await ctx.db
        .query("ragaDetections")
        .withIndex("by_user", (q) => q.eq("userId", user._id))
        .collect();
      
      for (const detection of detections) {
        await ctx.db.delete(detection._id);
      }
    }

    return { success: true };
  },
});

// Get detection by ID
export const getDetectionById = query({
  args: { detectionId: v.id("ragaDetections") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      return null;
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      return null;
    }

    const detection = await ctx.db.get(args.detectionId);
    if (!detection || detection.userId !== user._id) {
      return null;
    }

    // Get file information
    const file = await ctx.db.get(detection.audioFileId);
    
    return {
      ...detection,
      file,
    };
  },
});

// Export detection history
export const exportDetectionHistory = query({
  args: {
    format: v.union(v.literal("json"), v.literal("csv")),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new ConvexError("Not authenticated");
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      throw new ConvexError("User not found");
    }

    const detections = await ctx.db
      .query("ragaDetections")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .collect();

    if (args.format === "json") {
      return {
        format: "json",
        data: detections,
        exportDate: new Date().toISOString(),
      };
    } else {
      // CSV format
      const csvHeaders = [
        "Date",
        "Predicted Raga",
        "Confidence",
        "Processing Time (ms)",
        "Top Predictions"
      ];
      
      const csvRows = detections.map(detection => [
        new Date(detection.createdAt).toISOString(),
        detection.predictedRaga,
        detection.confidence,
        detection.processingTime,
        detection.topPredictions.map(p => `${p.raga}(${p.probability})`).join("; ")
      ]);

      const csvContent = [csvHeaders, ...csvRows]
        .map(row => row.map(cell => `"${cell}"`).join(","))
        .join("\n");

      return {
        format: "csv",
        data: csvContent,
        exportDate: new Date().toISOString(),
      };
    }
  },
});
