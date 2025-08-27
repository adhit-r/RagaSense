import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // User profiles and authentication
  users: defineTable({
    name: v.string(),
    email: v.string(),
    image: v.optional(v.string()),
    authId: v.string(),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_auth_id", ["authId"])
    .index("by_email", ["email"]),

  // Raga detection history
  ragaDetections: defineTable({
    userId: v.id("users"),
    audioFileId: v.id("files"),
    predictedRaga: v.string(),
    confidence: v.number(),
    topPredictions: v.array(v.object({
      raga: v.string(),
      probability: v.number(),
      confidence: v.string(),
    })),
    processingTime: v.number(),
    createdAt: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_created_at", ["createdAt"]),

  // Audio files storage
  files: defineTable({
    name: v.string(),
    size: v.number(),
    type: v.string(),
    userId: v.id("users"),
    storageId: v.string(),
    uploadedAt: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_storage_id", ["storageId"]),

  // Music generation requests
  musicGenerations: defineTable({
    userId: v.id("users"),
    prompt: v.string(),
    raga: v.string(),
    duration: v.number(),
    status: v.union(v.literal("pending"), v.literal("processing"), v.literal("completed"), v.literal("failed")),
    audioFileId: v.optional(v.id("files")),
    error: v.optional(v.string()),
    createdAt: v.number(),
    completedAt: v.optional(v.number()),
  })
    .index("by_user", ["userId"])
    .index("by_status", ["status"])
    .index("by_created_at", ["createdAt"]),

  // User preferences and settings
  userSettings: defineTable({
    userId: v.id("users"),
    theme: v.union(v.literal("light"), v.literal("dark"), v.literal("auto")),
    language: v.string(),
    notifications: v.boolean(),
    autoSave: v.boolean(),
    updatedAt: v.number(),
  })
    .index("by_user", ["userId"]),

  // Raga information and metadata
  ragas: defineTable({
    name: v.string(),
    alternateNames: v.array(v.string()),
    category: v.string(), // Carnatic, Hindustani
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
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_name", ["name"])
    .index("by_category", ["category"])
    .index("by_time_of_day", ["timeOfDay"]),

  // User favorites and collections
  userFavorites: defineTable({
    userId: v.id("users"),
    ragaId: v.id("ragas"),
    createdAt: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_raga", ["ragaId"])
    .index("by_user_and_raga", ["userId", "ragaId"]),

  // Analytics and usage statistics
  analytics: defineTable({
    eventType: v.string(),
    userId: v.optional(v.id("users")),
    metadata: v.any(),
    timestamp: v.number(),
  })
    .index("by_event_type", ["eventType"])
    .index("by_user", ["userId"])
    .index("by_timestamp", ["timestamp"]),
});
