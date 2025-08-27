import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Generate upload URL for audio files
export const generateUploadUrl = mutation({
  args: {},
  handler: async (ctx) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    return await ctx.storage.generateUploadUrl();
  },
});

// Get file URL
export const getFileUrl = query({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, args) => {
    return await ctx.storage.getUrl(args.storageId);
  },
});

// Delete file
export const deleteFile = mutation({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    await ctx.storage.delete(args.storageId);
  },
});

// Upload audio sample
export const uploadAudioSample = mutation({
  args: {
    storageId: v.id("_storage"),
    ragaId: v.id("ragas"),
    filename: v.string(),
    duration: v.number(),
    sampleRate: v.number(),
    format: v.string(),
    description: v.optional(v.string()),
    isPublic: v.boolean(),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;

    const audioSampleId = await ctx.db.insert("audioSamples", {
      userId,
      ragaId: args.ragaId,
      audioFileId: args.storageId,
      filename: args.filename,
      duration: args.duration,
      sampleRate: args.sampleRate,
      format: args.format,
      description: args.description,
      isPublic: args.isPublic,
      createdAt: Date.now(),
    });

    return audioSampleId;
  },
});

// Get user's audio samples
export const getUserAudioSamples = query({
  args: { isPublic: v.optional(v.boolean()) },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;
    let q = ctx.db.query("audioSamples").withIndex("by_user", (q) => 
      q.eq("userId", userId)
    );

    if (args.isPublic !== undefined) {
      q = q.withIndex("by_public", (q) => q.eq("isPublic", args.isPublic));
    }

    return await q.collect();
  },
});

// Get public audio samples
export const getPublicAudioSamples = query({
  args: { ragaId: v.optional(v.id("ragas")) },
  handler: async (ctx, args) => {
    let q = ctx.db.query("audioSamples").withIndex("by_public", (q) => 
      q.eq("isPublic", true)
    );

    if (args.ragaId) {
      q = q.withIndex("by_raga", (q) => q.eq("ragaId", args.ragaId));
    }

    return await q.collect();
  },
});

// Delete audio sample
export const deleteAudioSample = mutation({
  args: { audioSampleId: v.id("audioSamples") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const audioSample = await ctx.db.get(args.audioSampleId);
    if (!audioSample || audioSample.userId !== identity.subject) {
      throw new Error("Not found or not authorized");
    }

    // Delete the file from storage
    await ctx.storage.delete(audioSample.audioFileId);

    // Delete the database record
    await ctx.db.delete(args.audioSampleId);

    return args.audioSampleId;
  },
});
