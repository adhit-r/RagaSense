import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { ConvexError } from "convex/values";

// Get user's music generation history
export const getMusicGenerationHistory = query({
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

    const generations = await ctx.db
      .query("musicGenerations")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .paginate({
        numItems: limit,
        cursor: null,
      });

    // Get file information for completed generations
    const generationsWithFiles = await Promise.all(
      generations.page.map(async (generation) => {
        let file = null;
        if (generation.audioFileId) {
          file = await ctx.db.get(generation.audioFileId);
        }
        return {
          ...generation,
          file,
        };
      })
    );

    return generationsWithFiles;
  },
});

// Create music generation request
export const createMusicGeneration = mutation({
  args: {
    prompt: v.string(),
    raga: v.string(),
    duration: v.number(),
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

    return await ctx.db.insert("musicGenerations", {
      userId: user._id,
      prompt: args.prompt,
      raga: args.raga,
      duration: args.duration,
      status: "pending",
      createdAt: Date.now(),
    });
  },
});

// Update music generation status
export const updateMusicGenerationStatus = mutation({
  args: {
    generationId: v.id("musicGenerations"),
    status: v.union(v.literal("pending"), v.literal("processing"), v.literal("completed"), v.literal("failed")),
    audioFileId: v.optional(v.id("files")),
    error: v.optional(v.string()),
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

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== user._id) {
      throw new ConvexError("Generation not found or access denied");
    }

    const updates: any = {
      status: args.status,
    };

    if (args.status === "completed" && args.audioFileId) {
      updates.audioFileId = args.audioFileId;
      updates.completedAt = Date.now();
    } else if (args.status === "failed" && args.error) {
      updates.error = args.error;
      updates.completedAt = Date.now();
    }

    return await ctx.db.patch(args.generationId, updates);
  },
});

// Get music generation by ID
export const getMusicGenerationById = query({
  args: { generationId: v.id("musicGenerations") },
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

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== user._id) {
      return null;
    }

    // Get file information if completed
    let file = null;
    if (generation.audioFileId) {
      file = await ctx.db.get(generation.audioFileId);
    }

    return {
      ...generation,
      file,
    };
  },
});

// Get music generation statistics
export const getMusicGenerationStats = query({
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

    const generations = await ctx.db
      .query("musicGenerations")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();

    const totalGenerations = generations.length;
    const completedGenerations = generations.filter(g => g.status === "completed").length;
    const failedGenerations = generations.filter(g => g.status === "failed").length;
    const pendingGenerations = generations.filter(g => g.status === "pending").length;
    const processingGenerations = generations.filter(g => g.status === "processing").length;

    // Count by raga
    const ragaCounts = generations.reduce((acc, generation) => {
      acc[generation.raga] = (acc[generation.raga] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get most used ragas
    const topRagas = Object.entries(ragaCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([raga, count]) => ({ raga, count }));

    // Calculate average duration
    const avgDuration = generations.length > 0
      ? generations.reduce((sum, g) => sum + g.duration, 0) / generations.length
      : 0;

    return {
      totalGenerations,
      completedGenerations,
      failedGenerations,
      pendingGenerations,
      processingGenerations,
      successRate: totalGenerations > 0 ? (completedGenerations / totalGenerations) * 100 : 0,
      topRagas,
      avgDuration,
      recentGenerations: generations
        .sort((a, b) => b.createdAt - a.createdAt)
        .slice(0, 10),
    };
  },
});

// Delete music generation
export const deleteMusicGeneration = mutation({
  args: { generationId: v.id("musicGenerations") },
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

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== user._id) {
      throw new ConvexError("Generation not found or access denied");
    }

    // Delete associated audio file if exists
    if (generation.audioFileId) {
      await ctx.db.delete(generation.audioFileId);
    }

    await ctx.db.delete(args.generationId);
    return { success: true };
  },
});

// Get pending generations (for background processing)
export const getPendingGenerations = query({
  args: {},
  handler: async (ctx) => {
    // This could be used by a background worker
    // For now, we'll require authentication
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      return [];
    }

    return await ctx.db
      .query("musicGenerations")
      .withIndex("by_status", (q) => q.eq("status", "pending"))
      .collect();
  },
});

// Export music generation history
export const exportMusicGenerationHistory = query({
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

    const generations = await ctx.db
      .query("musicGenerations")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .collect();

    if (args.format === "json") {
      return {
        format: "json",
        data: generations,
        exportDate: new Date().toISOString(),
      };
    } else {
      // CSV format
      const csvHeaders = [
        "Date",
        "Prompt",
        "Raga",
        "Duration (seconds)",
        "Status",
        "Error"
      ];
      
      const csvRows = generations.map(generation => [
        new Date(generation.createdAt).toISOString(),
        generation.prompt,
        generation.raga,
        generation.duration,
        generation.status,
        generation.error || ""
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
