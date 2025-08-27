import { query, mutation } from "./_generated/server";
import { v } from "convex/values";

// Get all ragas
export const getAll = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("ragas").collect();
  },
});

// Get raga by name
export const getByName = query({
  args: { name: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_name", (q) => q.eq("name", args.name))
      .first();
  },
});

// Get ragas by tradition
export const getByTradition = query({
  args: { tradition: v.union(v.literal("Hindustani"), v.literal("Carnatic")) },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_tradition", (q) => q.eq("tradition", args.tradition))
      .collect();
  },
});

// Search ragas
export const search = query({
  args: { 
    query: v.string(),
    tradition: v.optional(v.union(v.literal("Hindustani"), v.literal("Carnatic")))
  },
  handler: async (ctx, args) => {
    let q = ctx.db.query("ragas");
    
    if (args.tradition) {
      q = q.withIndex("by_tradition", (q) => q.eq("tradition", args.tradition));
    }
    
    const ragas = await q.collect();
    
    // Filter by search query
    return ragas.filter(raga => 
      raga.name.toLowerCase().includes(args.query.toLowerCase()) ||
      raga.description?.toLowerCase().includes(args.query.toLowerCase()) ||
      raga.alternateNames?.some(name => 
        name.toLowerCase().includes(args.query.toLowerCase())
      )
    );
  },
});

// Get ragas by mood
export const getByMood = query({
  args: { mood: v.string() },
  handler: async (ctx, args) => {
    const ragas = await ctx.db.query("ragas").collect();
    
    return ragas.filter(raga => 
      raga.mood?.some(m => m.toLowerCase().includes(args.mood.toLowerCase()))
    );
  },
});

// Get ragas by time of day
export const getByTime = query({
  args: { time: v.string() },
  handler: async (ctx, args) => {
    const ragas = await ctx.db.query("ragas").collect();
    
    return ragas.filter(raga => 
      raga.time?.some(t => t.toLowerCase().includes(args.time.toLowerCase()))
    );
  },
});

// Get ragas by thaat (Hindustani)
export const getByThaat = query({
  args: { thaat: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_thaat", (q) => q.eq("thaat", args.thaat))
      .collect();
  },
});

// Get ragas by melakarta number (Carnatic)
export const getByMelakarta = query({
  args: { melakartaNumber: v.number() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_melakarta", (q) => q.eq("melakartaNumber", args.melakartaNumber))
      .collect();
  },
});

// Get suggested ragas by mood (for music generation)
export const getSuggestedByMood = query({
  args: { 
    mood: v.union(
      v.literal("peaceful"), 
      v.literal("joyful"), 
      v.literal("romantic"), 
      v.literal("energetic"), 
      v.literal("melancholic")
    )
  },
  handler: async (ctx, args) => {
    const moodRagaMapping = {
      peaceful: ['Yaman', 'Bhairav', 'Malkauns', 'Bageshri', 'Darbari'],
      joyful: ['Bilawal', 'Kafi', 'Bhairavi', 'Des', 'Khamaj'],
      romantic: ['Khamaj', 'Des', 'Bageshri', 'Pilu', 'Tilak Kamod'],
      energetic: ['Jog', 'Hansdhwani', 'Shivaranjani', 'Durga', 'Miyan Malhar'],
      melancholic: ['Darbari', 'Marwa', 'Puriya', 'Malkauns', 'Bhairav']
    };

    const suggestedRagas = moodRagaMapping[args.mood];
    const ragas = await ctx.db.query("ragas").collect();
    
    return ragas.filter(raga => suggestedRagas.includes(raga.name));
  },
});

// Get suggested ragas by theme (for music generation)
export const getSuggestedByTheme = query({
  args: { 
    theme: v.union(
      v.literal("spiritual"), 
      v.literal("cultural"), 
      v.literal("contemporary"), 
      v.literal("educational")
    )
  },
  handler: async (ctx, args) => {
    const themeRagaMapping = {
      spiritual: ['Bhairav', 'Yaman', 'Malkauns', 'Bageshri', 'Darbari'],
      cultural: ['Bilawal', 'Des', 'Kafi', 'Bhairavi', 'Khamaj'],
      contemporary: ['Khamaj', 'Des', 'Pilu', 'Tilak Kamod', 'Hansdhwani'],
      educational: ['Bilawal', 'Yaman', 'Kafi', 'Bhairav', 'Khamaj']
    };

    const suggestedRagas = themeRagaMapping[args.theme];
    const ragas = await ctx.db.query("ragas").collect();
    
    return ragas.filter(raga => suggestedRagas.includes(raga.name));
  },
});

// Add a new raga (admin function)
export const addRaga = mutation({
  args: {
    name: v.string(),
    tradition: v.union(v.literal("Hindustani"), v.literal("Carnatic")),
    arohana: v.array(v.string()),
    avarohana: v.array(v.string()),
    description: v.optional(v.string()),
    vadi: v.optional(v.string()),
    samvadi: v.optional(v.string()),
    time: v.optional(v.array(v.string())),
    mood: v.optional(v.array(v.string())),
    thaat: v.optional(v.string()),
    melakartaNumber: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    // Check if raga already exists
    const existing = await ctx.db
      .query("ragas")
      .withIndex("by_name", (q) => q.eq("name", args.name))
      .first();

    if (existing) {
      throw new Error("Raga already exists");
    }

    const ragaId = await ctx.db.insert("ragas", {
      name: args.name,
      tradition: args.tradition,
      arohana: args.arohana,
      avarohana: args.avarohana,
      description: args.description,
      vadi: args.vadi,
      samvadi: args.samvadi,
      time: args.time,
      mood: args.mood,
      thaat: args.thaat,
      melakartaNumber: args.melakartaNumber,
    });

    return ragaId;
  },
});

// Update raga
export const updateRaga = mutation({
  args: {
    ragaId: v.id("ragas"),
    updates: v.object({
      name: v.optional(v.string()),
      description: v.optional(v.string()),
      arohana: v.optional(v.array(v.string())),
      avarohana: v.optional(v.array(v.string())),
      vadi: v.optional(v.string()),
      samvadi: v.optional(v.string()),
      time: v.optional(v.array(v.string())),
      mood: v.optional(v.array(v.string())),
    }),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    await ctx.db.patch(args.ragaId, args.updates);
    return args.ragaId;
  },
});

// Delete raga
export const deleteRaga = mutation({
  args: { ragaId: v.id("ragas") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    await ctx.db.delete(args.ragaId);
    return args.ragaId;
  },
});
