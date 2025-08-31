import { v } from "convex/values";
import { query } from "./_generated/server";

export const getFile = query({
  args: { fileId: v.id("files") },
  handler: async (ctx, args) => {
    const file = await ctx.db.get(args.fileId);
    if (!file) {
      throw new Error("File not found");
    }
    
    // Generate a URL for the file (this would typically use Convex file storage)
    // For now, return a placeholder URL
    return {
      ...file,
      url: `/api/files/${file.storageId}`, // This would be the actual file URL
    };
  },
});

export const getUserFiles = query({
  args: {},
  handler: async (ctx, args) => {
    if (!ctx.auth.userId) {
      return [];
    }
    
    return await ctx.db
      .query("files")
      .filter((q) => q.eq(q.field("userId"), ctx.auth.userId))
      .order("desc")
      .collect();
  },
});
