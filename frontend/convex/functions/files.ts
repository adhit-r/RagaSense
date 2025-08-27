import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { ConvexError } from "convex/values";

// Get user's files
export const getUserFiles = query({
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

    const files = await ctx.db
      .query("files")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .paginate({
        numItems: limit,
        cursor: null,
      });

    return files.page;
  },
});

// Get file by ID
export const getFileById = query({
  args: { fileId: v.id("files") },
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

    const file = await ctx.db.get(args.fileId);
    if (!file || file.userId !== user._id) {
      return null;
    }

    return file;
  },
});

// Create file record
export const createFile = mutation({
  args: {
    name: v.string(),
    size: v.number(),
    type: v.string(),
    storageId: v.string(),
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

    return await ctx.db.insert("files", {
      name: args.name,
      size: args.size,
      type: args.type,
      userId: user._id,
      storageId: args.storageId,
      uploadedAt: Date.now(),
    });
  },
});

// Update file record
export const updateFile = mutation({
  args: {
    fileId: v.id("files"),
    name: v.optional(v.string()),
    size: v.optional(v.number()),
    type: v.optional(v.string()),
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

    const file = await ctx.db.get(args.fileId);
    if (!file || file.userId !== user._id) {
      throw new ConvexError("File not found or access denied");
    }

    const { fileId, ...updates } = args;
    return await ctx.db.patch(fileId, updates);
  },
});

// Delete file
export const deleteFile = mutation({
  args: { fileId: v.id("files") },
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

    const file = await ctx.db.get(args.fileId);
    if (!file || file.userId !== user._id) {
      throw new ConvexError("File not found or access denied");
    }

    await ctx.db.delete(args.fileId);
    return { success: true };
  },
});

// Get file statistics
export const getFileStats = query({
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

    const files = await ctx.db
      .query("files")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();

    const totalFiles = files.length;
    const totalSize = files.reduce((sum, file) => sum + file.size, 0);
    
    // Group by file type
    const typeCounts = files.reduce((acc, file) => {
      const type = file.type.split('/')[0]; // audio, image, etc.
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get recent uploads
    const recentFiles = files
      .sort((a, b) => b.uploadedAt - a.uploadedAt)
      .slice(0, 10);

    return {
      totalFiles,
      totalSize,
      typeCounts,
      recentFiles,
    };
  },
});

// Search files
export const searchFiles = query({
  args: { query: v.string() },
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

    const files = await ctx.db
      .query("files")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();

    const query = args.query.toLowerCase();
    return files.filter(file => 
      file.name.toLowerCase().includes(query) ||
      file.type.toLowerCase().includes(query)
    );
  },
});

// Get files by type
export const getFilesByType = query({
  args: { type: v.string() },
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

    const files = await ctx.db
      .query("files")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();

    return files.filter(file => file.type.startsWith(args.type));
  },
});
