import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { ConvexError } from "convex/values";

// Get current user
export const getCurrentUser = query({
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

    return user;
  },
});

// Create or update user profile
export const createOrUpdateUser = mutation({
  args: {
    name: v.string(),
    email: v.string(),
    image: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new ConvexError("Not authenticated");
    }

    const existingUser = await ctx.db
      .query("users")
      .withIndex("by_auth_id", (q) => q.eq("authId", identity.subject))
      .unique();

    if (existingUser) {
      // Update existing user
      return await ctx.db.patch(existingUser._id, {
        name: args.name,
        email: args.email,
        image: args.image,
        updatedAt: Date.now(),
      });
    } else {
      // Create new user
      return await ctx.db.insert("users", {
        name: args.name,
        email: args.email,
        image: args.image,
        authId: identity.subject,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      });
    }
  },
});

// Get user settings
export const getUserSettings = query({
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

    const settings = await ctx.db
      .query("userSettings")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .unique();

    return settings;
  },
});

// Update user settings
export const updateUserSettings = mutation({
  args: {
    theme: v.optional(v.union(v.literal("light"), v.literal("dark"), v.literal("auto"))),
    language: v.optional(v.string()),
    notifications: v.optional(v.boolean()),
    autoSave: v.optional(v.boolean()),
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

    const existingSettings = await ctx.db
      .query("userSettings")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .unique();

    if (existingSettings) {
      // Update existing settings
      return await ctx.db.patch(existingSettings._id, {
        ...args,
        updatedAt: Date.now(),
      });
    } else {
      // Create new settings with defaults
      return await ctx.db.insert("userSettings", {
        userId: user._id,
        theme: args.theme ?? "auto",
        language: args.language ?? "en",
        notifications: args.notifications ?? true,
        autoSave: args.autoSave ?? true,
        updatedAt: Date.now(),
      });
    }
  },
});

// Delete user account
export const deleteUser = mutation({
  args: {},
  handler: async (ctx) => {
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

    // Delete all user data
    await ctx.db.delete(user._id);

    // Delete user settings
    const settings = await ctx.db
      .query("userSettings")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .unique();
    
    if (settings) {
      await ctx.db.delete(settings._id);
    }

    // Delete user favorites
    const favorites = await ctx.db
      .query("userFavorites")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();
    
    for (const favorite of favorites) {
      await ctx.db.delete(favorite._id);
    }

    // Delete user files
    const files = await ctx.db
      .query("files")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();
    
    for (const file of files) {
      await ctx.db.delete(file._id);
    }

    // Delete raga detections
    const detections = await ctx.db
      .query("ragaDetections")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();
    
    for (const detection of detections) {
      await ctx.db.delete(detection._id);
    }

    // Delete music generations
    const generations = await ctx.db
      .query("musicGenerations")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();
    
    for (const generation of generations) {
      await ctx.db.delete(generation._id);
    }

    return { success: true };
  },
});
