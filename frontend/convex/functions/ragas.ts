import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { ConvexError } from "convex/values";

// Get all ragas
export const getAllRagas = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("ragas").collect();
  },
});

// Get raga by name
export const getRagaByName = query({
  args: { name: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_name", (q) => q.eq("name", args.name))
      .unique();
  },
});

// Get ragas by category
export const getRagasByCategory = query({
  args: { category: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_category", (q) => q.eq("category", args.category))
      .collect();
  },
});

// Get ragas by time of day
export const getRagasByTimeOfDay = query({
  args: { timeOfDay: v.union(v.literal("morning"), v.literal("afternoon"), v.literal("evening"), v.literal("night")) },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_time_of_day", (q) => q.eq("timeOfDay", args.timeOfDay))
      .collect();
  },
});

// Search ragas
export const searchRagas = query({
  args: { query: v.string() },
  handler: async (ctx, args) => {
    const allRagas = await ctx.db.query("ragas").collect();
    const query = args.query.toLowerCase();
    
    return allRagas.filter(raga => 
      raga.name.toLowerCase().includes(query) ||
      raga.alternateNames.some(name => name.toLowerCase().includes(query)) ||
      raga.description.toLowerCase().includes(query)
    );
  },
});

// Create new raga (admin only)
export const createRaga = mutation({
  args: {
    name: v.string(),
    alternateNames: v.array(v.string()),
    category: v.string(),
    timeOfDay: v.union(v.literal("morning"), v.literal("afternoon"), v.literal("evening"), v.literal("night")),
    season: v.optional(v.string()),
    mood: v.array(v.string()),
    description: v.string(),
    notes: v.array(v.string()),
    arohana: v.array(v.string()),
    avarohana: v.array(v.string()),
    pakad: v.optional(v.string()),
    vadi: v.optional(v.string()),
    samvadi: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    // TODO: Add admin check
    return await ctx.db.insert("ragas", {
      ...args,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    });
  },
});

// Update raga (admin only)
export const updateRaga = mutation({
  args: {
    ragaId: v.id("ragas"),
    name: v.optional(v.string()),
    alternateNames: v.optional(v.array(v.string())),
    category: v.optional(v.string()),
    timeOfDay: v.optional(v.union(v.literal("morning"), v.literal("afternoon"), v.literal("evening"), v.literal("night"))),
    season: v.optional(v.string()),
    mood: v.optional(v.array(v.string())),
    description: v.optional(v.string()),
    notes: v.optional(v.array(v.string())),
    arohana: v.optional(v.array(v.string())),
    avarohana: v.optional(v.array(v.string())),
    pakad: v.optional(v.string()),
    vadi: v.optional(v.string()),
    samvadi: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    // TODO: Add admin check
    const { ragaId, ...updates } = args;
    return await ctx.db.patch(ragaId, {
      ...updates,
      updatedAt: Date.now(),
    });
  },
});

// Delete raga (admin only)
export const deleteRaga = mutation({
  args: { ragaId: v.id("ragas") },
  handler: async (ctx, args) => {
    // TODO: Add admin check
    await ctx.db.delete(args.ragaId);
    return { success: true };
  },
});

// Get user favorites
export const getUserFavorites = query({
  args: {},
  handler: async (ctx) => {
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

    const favorites = await ctx.db
      .query("userFavorites")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();

    // Get the actual raga data for each favorite
    const ragas = await Promise.all(
      favorites.map(async (favorite) => {
        const raga = await ctx.db.get(favorite.ragaId);
        return raga;
      })
    );

    return ragas.filter(Boolean);
  },
});

// Add raga to favorites
export const addToFavorites = mutation({
  args: { ragaId: v.id("ragas") },
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

    // Check if already favorited
    const existing = await ctx.db
      .query("userFavorites")
      .withIndex("by_user_and_raga", (q) => 
        q.eq("userId", user._id).eq("ragaId", args.ragaId)
      )
      .unique();

    if (existing) {
      throw new ConvexError("Raga already in favorites");
    }

    return await ctx.db.insert("userFavorites", {
      userId: user._id,
      ragaId: args.ragaId,
      createdAt: Date.now(),
    });
  },
});

// Remove raga from favorites
export const removeFromFavorites = mutation({
  args: { ragaId: v.id("ragas") },
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

    const favorite = await ctx.db
      .query("userFavorites")
      .withIndex("by_user_and_raga", (q) => 
        q.eq("userId", user._id).eq("ragaId", args.ragaId)
      )
      .unique();

    if (!favorite) {
      throw new ConvexError("Raga not in favorites");
    }

    await ctx.db.delete(favorite._id);
    return { success: true };
  },
});

// Check if raga is favorited
export const isFavorited = query({
  args: { ragaId: v.id("ragas") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      return false;
    }

    const user = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (!user) {
      return false;
    }

    const favorite = await ctx.db
      .query("userFavorites")
      .withIndex("by_user_and_raga", (q) => 
        q.eq("userId", user._id).eq("ragaId", args.ragaId)
      )
      .unique();

    return !!favorite;
  },
});
