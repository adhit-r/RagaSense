
// convex/generateMusic.ts
import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

export const generateRagaMusic = mutation({
  args: {
    ragaName: v.string(),
    duration: v.optional(v.number()),
    style: v.optional(v.string()), // "alapana", "kriti", "varnam", "thillana"
  },
  handler: async (ctx, args) => {
    // This would integrate with the MusicGen model
    // For now, return a placeholder
    
    const result = {
      ragaName: args.ragaName,
      duration: args.duration || 30,
      style: args.style || "alapana",
      status: "generating",
      audioUrl: null,
      timestamp: new Date().toISOString(),
    };
    
    // Store generation request
    const generationId = await ctx.db.insert("musicGenerations", {
      ragaName: args.ragaName,
      duration: args.duration || 30,
      style: args.style || "alapana",
      status: "pending",
      createdAt: new Date().toISOString(),
    });
    
    return { generationId, result };
  },
});

export const getMusicGeneration = query({
  args: { id: v.id("musicGenerations") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.id);
  },
});

export const getAvailableRagas = query({
  args: {},
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .filter((q) => q.eq(q.field("tradition"), "carnatic"))
      .collect();
  },
});

export const getMusicHistory = query({
  args: {},
  handler: async (ctx, args) => {
    if (!ctx.auth.userId) {
      return [];
    }
    
    return await ctx.db
      .query("musicGenerations")
      .filter((q) => q.eq(q.field("userId"), ctx.auth.userId))
      .order("desc")
      .collect();
  },
});
