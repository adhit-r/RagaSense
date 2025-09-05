// convex/getRagas.ts
import { v } from "convex/values";
import { query } from "./_generated/server";

export const getRagas = query({
  args: {
    tradition: v.optional(v.string()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    let query = ctx.db.query("ragas");
    
    if (args.tradition) {
      query = query.filter((q) => q.eq(q.field("tradition"), args.tradition));
    }
    
    if (args.limit) {
      query = query.take(args.limit);
    }
    
    return await query.collect();
  },
});

export const searchRagas = query({
  args: { searchTerm: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .filter((q) => q.gte(q.field("name"), args.searchTerm))
      .filter((q) => q.lt(q.field("name"), args.searchTerm + "\uffff"))
      .collect();
  },
});
