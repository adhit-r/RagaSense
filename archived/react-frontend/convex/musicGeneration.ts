import { mutation, query, action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";

// Start music generation
export const startGeneration = mutation({
  args: {
    request: v.object({
      musicType: v.union(v.literal("instrumental"), v.literal("vocal")),
      instruments: v.optional(v.object({
        primary: v.string(),
        secondary: v.optional(v.array(v.string())),
        ensemble: v.optional(v.boolean()),
      })),
      voice: v.optional(v.object({
        gender: v.union(v.literal("male"), v.literal("female")),
        pitch: v.union(v.literal("high"), v.literal("medium"), v.literal("low")),
        style: v.union(v.literal("classical"), v.literal("semi-classical"), v.literal("devotional")),
      })),
      mood: v.object({
        category: v.union(
          v.literal("peaceful"), 
          v.literal("joyful"), 
          v.literal("romantic"), 
          v.literal("energetic"), 
          v.literal("melancholic")
        ),
        intensity: v.number(),
        suggestedRagas: v.array(v.string()),
        selectedRaga: v.optional(v.string()),
      }),
      theme: v.object({
        category: v.union(
          v.literal("spiritual"), 
          v.literal("cultural"), 
          v.literal("contemporary"), 
          v.literal("educational")
        ),
        subcategory: v.optional(v.string()),
        context: v.optional(v.string()),
      }),
      duration: v.number(), // in seconds
      tempo: v.optional(v.number()), // BPM
      key: v.optional(v.string()),
    }),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;

    const generationId = await ctx.db.insert("musicGenerations", {
      userId,
      request: args.request,
      status: "processing",
      progress: 0,
      isFavorite: false,
      createdAt: Date.now(),
    });

    // Trigger AI generation (simulated for now)
    await ctx.scheduler.runAfter(0, api.musicGeneration.processGeneration, {
      generationId,
      request: args.request,
    });

    return generationId;
  },
});

// Process music generation (simulated)
export const processGeneration = action({
  args: {
    generationId: v.id("musicGenerations"),
    request: v.any(),
  },
  handler: async (ctx, args) => {
    // Simulate generation progress
    for (let progress = 10; progress <= 100; progress += 10) {
      await ctx.runMutation(api.musicGeneration.updateProgress, {
        generationId: args.generationId,
        progress,
      });
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Mark as completed
    await ctx.runMutation(api.musicGeneration.completeGeneration, {
      generationId: args.generationId,
      metadata: {
        raga: args.request.mood.selectedRaga || "Yaman",
        instruments: args.request.instruments ? [args.request.instruments.primary] : ["Sitar"],
        duration: args.request.duration,
        tempo: args.request.tempo || 80,
        key: args.request.key || "C",
        mood: args.request.mood.category,
        theme: args.request.theme.category,
      },
    });
  },
});

// Update generation progress
export const updateProgress = mutation({
  args: {
    generationId: v.id("musicGenerations"),
    progress: v.number(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.generationId, {
      progress: args.progress,
    });
  },
});

// Complete generation
export const completeGeneration = mutation({
  args: {
    generationId: v.id("musicGenerations"),
    metadata: v.object({
      raga: v.string(),
      instruments: v.array(v.string()),
      duration: v.number(),
      tempo: v.number(),
      key: v.string(),
      mood: v.string(),
      theme: v.string(),
    }),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.generationId, {
      status: "completed",
      progress: 100,
      metadata: args.metadata,
      completedAt: Date.now(),
    });
  },
});

// Get user's generation history
export const getUserHistory = query({
  args: { 
    isFavorite: v.optional(v.boolean()),
    status: v.optional(v.union(
      v.literal("processing"),
      v.literal("completed"),
      v.literal("failed")
    ))
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;
    let q = ctx.db.query("musicGenerations").withIndex("by_user", (q) => 
      q.eq("userId", userId)
    );

    if (args.isFavorite !== undefined) {
      q = q.withIndex("by_user_favorites", (q) => 
        q.eq("userId", userId).eq("isFavorite", args.isFavorite)
      );
    }

    const generations = await q.collect();

    // Filter by status if specified
    if (args.status) {
      return generations.filter(g => g.status === args.status);
    }

    return generations;
  },
});

// Get generation by ID
export const getGeneration = query({
  args: { generationId: v.id("musicGenerations") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== identity.subject) {
      throw new Error("Not found or not authorized");
    }

    return generation;
  },
});

// Toggle favorite
export const toggleFavorite = mutation({
  args: { generationId: v.id("musicGenerations") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== identity.subject) {
      throw new Error("Not found or not authorized");
    }

    await ctx.db.patch(args.generationId, {
      isFavorite: !generation.isFavorite,
    });

    return !generation.isFavorite;
  },
});

// Delete generation
export const deleteGeneration = mutation({
  args: { generationId: v.id("musicGenerations") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== identity.subject) {
      throw new Error("Not found or not authorized");
    }

    await ctx.db.delete(args.generationId);
    return args.generationId;
  },
});

// Get generation statistics
export const getStats = query({
  args: {},
  handler: async (ctx) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;
    const generations = await ctx.db
      .query("musicGenerations")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();

    const total = generations.length;
    const completed = generations.filter(g => g.status === "completed").length;
    const processing = generations.filter(g => g.status === "processing").length;
    const failed = generations.filter(g => g.status === "failed").length;
    const favorites = generations.filter(g => g.isFavorite).length;

    return {
      total,
      completed,
      processing,
      failed,
      favorites,
    };
  },
});
