// convex/classifyRaga.ts
import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

export const classifyRaga = mutation({
  args: {
    audioUrl: v.string(),
    tradition: v.optional(v.string()), // "carnatic", "hindustani", "auto"
  },
  handler: async (ctx, args) => {
    // This would integrate with the hybrid classifier
    // For now, return a placeholder
    
    const result = {
      tradition: args.tradition || "auto",
      predictedRaga: "unknown",
      confidence: 0.0,
      modelSource: "hybrid",
      timestamp: new Date().toISOString(),
    };
    
    // Store classification result
    const classificationId = await ctx.db.insert("classifications", {
      audioUrl: args.audioUrl,
      result: result,
      createdAt: new Date().toISOString(),
    });
    
    return { classificationId, result };
  },
});

export const getClassification = query({
  args: { id: v.id("classifications") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.id);
  },
});
