import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// Track analytics event
export const trackEvent = mutation({
  args: {
    eventType: v.string(),
    metadata: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    const userId = identity ? identity.subject : null;

    return await ctx.db.insert("analytics", {
      eventType: args.eventType,
      userId: userId,
      metadata: args.metadata || {},
      timestamp: Date.now(),
    });
  },
});

// Get analytics for a specific event type
export const getEventAnalytics = query({
  args: {
    eventType: v.string(),
    startDate: v.optional(v.number()),
    endDate: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const startDate = args.startDate || Date.now() - 30 * 24 * 60 * 60 * 1000; // 30 days ago
    const endDate = args.endDate || Date.now();

    const events = await ctx.db
      .query("analytics")
      .withIndex("by_event_type", (q) => q.eq("eventType", args.eventType))
      .filter((q) => 
        q.and(
          q.gte(q.field("timestamp"), startDate),
          q.lte(q.field("timestamp"), endDate)
        )
      )
      .collect();

    return {
      eventType: args.eventType,
      totalEvents: events.length,
      events: events,
      startDate: new Date(startDate).toISOString(),
      endDate: new Date(endDate).toISOString(),
    };
  },
});

// Get user analytics
export const getUserAnalytics = query({
  args: {
    startDate: v.optional(v.number()),
    endDate: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      return null;
    }

    const startDate = args.startDate || Date.now() - 30 * 24 * 60 * 60 * 1000; // 30 days ago
    const endDate = args.endDate || Date.now();

    const events = await ctx.db
      .query("analytics")
      .withIndex("by_user", (q) => q.eq("userId", identity.subject))
      .filter((q) => 
        q.and(
          q.gte(q.field("timestamp"), startDate),
          q.lte(q.field("timestamp"), endDate)
        )
      )
      .collect();

    // Group events by type
    const eventCounts = events.reduce((acc, event) => {
      acc[event.eventType] = (acc[event.eventType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get daily activity
    const dailyActivity = events.reduce((acc, event) => {
      const date = new Date(event.timestamp).toDateString();
      acc[date] = (acc[date] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalEvents: events.length,
      eventCounts,
      dailyActivity,
      events: events.sort((a, b) => b.timestamp - a.timestamp).slice(0, 50),
      startDate: new Date(startDate).toISOString(),
      endDate: new Date(endDate).toISOString(),
    };
  },
});

// Get global analytics (admin only)
export const getGlobalAnalytics = query({
  args: {
    startDate: v.optional(v.number()),
    endDate: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    // TODO: Add admin check
    const startDate = args.startDate || Date.now() - 30 * 24 * 60 * 60 * 1000; // 30 days ago
    const endDate = args.endDate || Date.now();

    const events = await ctx.db
      .query("analytics")
      .withIndex("by_timestamp", (q) => 
        q.and(
          q.gte(q.field("timestamp"), startDate),
          q.lte(q.field("timestamp"), endDate)
        )
      )
      .collect();

    // Group events by type
    const eventCounts = events.reduce((acc, event) => {
      acc[event.eventType] = (acc[event.eventType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get unique users
    const uniqueUsers = new Set(events.filter(e => e.userId).map(e => e.userId)).size;

    // Get daily activity
    const dailyActivity = events.reduce((acc, event) => {
      const date = new Date(event.timestamp).toDateString();
      acc[date] = (acc[date] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get hourly activity
    const hourlyActivity = events.reduce((acc, event) => {
      const hour = new Date(event.timestamp).getHours();
      acc[hour] = (acc[hour] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);

    return {
      totalEvents: events.length,
      uniqueUsers,
      eventCounts,
      dailyActivity,
      hourlyActivity,
      startDate: new Date(startDate).toISOString(),
      endDate: new Date(endDate).toISOString(),
    };
  },
});

// Track specific events
export const trackRagaDetection = mutation({
  args: {
    raga: v.string(),
    confidence: v.number(),
    processingTime: v.number(),
    success: v.boolean(),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    const userId = identity ? identity.subject : null;

    return await ctx.db.insert("analytics", {
      eventType: "raga_detection",
      userId: userId,
      metadata: {
        raga: args.raga,
        confidence: args.confidence,
        processingTime: args.processingTime,
        success: args.success,
      },
      timestamp: Date.now(),
    });
  },
});

export const trackMusicGeneration = mutation({
  args: {
    raga: v.string(),
    duration: v.number(),
    success: v.boolean(),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    const userId = identity ? identity.subject : null;

    return await ctx.db.insert("analytics", {
      eventType: "music_generation",
      userId: userId,
      metadata: {
        raga: args.raga,
        duration: args.duration,
        success: args.success,
        error: args.error,
      },
      timestamp: Date.now(),
    });
  },
});

export const trackUserAction = mutation({
  args: {
    action: v.string(),
    page: v.optional(v.string()),
    metadata: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    const userId = identity ? identity.subject : null;

    return await ctx.db.insert("analytics", {
      eventType: "user_action",
      userId: userId,
      metadata: {
        action: args.action,
        page: args.page,
        ...args.metadata,
      },
      timestamp: Date.now(),
    });
  },
});

export const trackError = mutation({
  args: {
    error: v.string(),
    page: v.optional(v.string()),
    metadata: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    const userId = identity ? identity.subject : null;

    return await ctx.db.insert("analytics", {
      eventType: "error",
      userId: userId,
      metadata: {
        error: args.error,
        page: args.page,
        ...args.metadata,
      },
      timestamp: Date.now(),
    });
  },
});
