import { defineSchema, defineTable } from "convex/schema";
import { v } from "convex/values";

export default defineSchema({
  // Ragas table - replacing PostgreSQL ragas
  ragas: defineTable({
    name: v.string(),
    alternateNames: v.optional(v.array(v.string())),
    tradition: v.union(v.literal("Hindustani"), v.literal("Carnatic")),
    
    // Scale information
    arohana: v.array(v.string()), // Ascending scale
    avarohana: v.array(v.string()), // Descending scale
    characteristicPhrases: v.optional(v.array(v.string())),
    
    // Characteristics
    vadi: v.optional(v.string()), // King note
    samvadi: v.optional(v.string()), // Queen note
    varjyaSwaras: v.optional(v.array(v.string())), // Omitted notes
    jati: v.optional(v.string()), // Classification
    
    // Performance context
    time: v.optional(v.array(v.string())), // Time of day
    season: v.optional(v.array(v.string())), // Season
    
    // Emotional content
    rasa: v.optional(v.array(v.string())), // Emotional essence
    mood: v.optional(v.array(v.string())), // Mood/feeling
    
    // Additional metadata
    description: v.optional(v.string()),
    history: v.optional(v.string()),
    notableCompositions: v.optional(v.array(v.string())),
    
    // Audio features (for ML model)
    audioFeatures: v.optional(v.any()),
    pitchDistribution: v.optional(v.any()),
    tonicFrequency: v.optional(v.number()),
    arohaPatterns: v.optional(v.any()),
    avarohaPatterns: v.optional(v.any()),
    pakad: v.optional(v.string()), // Characteristic catch phrase
    practiceExercises: v.optional(v.array(v.string())),
    
    // Tradition-specific fields
    thaat: v.optional(v.string()), // Hindustani classification
    timePeriod: v.optional(v.string()),
    regionalStyle: v.optional(v.array(v.string())),
    melakartaNumber: v.optional(v.number()), // Carnatic classification
    carnaticEquivalent: v.optional(v.string()),
    hindustaniEquivalent: v.optional(v.string()),
    janakaRaga: v.optional(v.string()),
    janyaRagas: v.optional(v.array(v.string())),
    chakra: v.optional(v.string()),
    
    // Legacy fields for compatibility
    icon: v.optional(v.string()),
    melakartaName: v.optional(v.string()),
    stats: v.optional(v.any()),
    info: v.optional(v.any()),
    songs: v.optional(v.array(v.string())),
    keyboard: v.optional(v.any()),
  })
    .index("by_name", ["name"])
    .index("by_tradition", ["tradition"])
    .index("by_thaat", ["thaat"])
    .index("by_melakarta", ["melakartaNumber"]),

  // Music Generation History
  musicGenerations: defineTable({
    userId: v.id("users"),
    request: v.object({
      musicType: v.union(v.literal("instrumental"), v.literal("vocal")),
      instruments: v.optional(v.object({
        primary: v.string(),
        secondary: v.optional(v.array(v.string())),
        ensemble: v.optional(v.boolean()),
      })),
      voice: v.optional(v.object({
        gender: v.union(v.literal("male"), v.literal("female")),
        pitch: v.union(v.literal("high"), v.literal("medium"), v.literal("low")),
        style: v.union(v.literal("classical"), v.literal("semi-classical"), v.literal("devotional")),
      })),
      mood: v.object({
        category: v.union(
          v.literal("peaceful"), 
          v.literal("joyful"), 
          v.literal("romantic"), 
          v.literal("energetic"), 
          v.literal("melancholic")
        ),
        intensity: v.number(),
        suggestedRagas: v.array(v.string()),
        selectedRaga: v.optional(v.string()),
      }),
      theme: v.object({
        category: v.union(
          v.literal("spiritual"), 
          v.literal("cultural"), 
          v.literal("contemporary"), 
          v.literal("educational")
        ),
        subcategory: v.optional(v.string()),
        context: v.optional(v.string()),
      }),
      duration: v.number(), // in seconds
      tempo: v.optional(v.number()), // BPM
      key: v.optional(v.string()),
    }),
    status: v.union(
      v.literal("processing"),
      v.literal("completed"),
      v.literal("failed")
    ),
    audioFileId: v.optional(v.id("_storage")),
    metadata: v.optional(v.object({
      raga: v.string(),
      instruments: v.array(v.string()),
      duration: v.number(),
      tempo: v.number(),
      key: v.string(),
      mood: v.string(),
      theme: v.string(),
    })),
    progress: v.optional(v.number()), // 0-100
    error: v.optional(v.string()),
    isFavorite: v.boolean(),
    createdAt: v.number(),
    completedAt: v.optional(v.number()),
  })
    .index("by_user", ["userId"])
    .index("by_status", ["status"])
    .index("by_user_favorites", ["userId", "isFavorite"])
    .index("by_created_at", ["createdAt"]),

  // Users (handled by Convex Auth)
  users: defineTable({
    email: v.string(),
    name: v.optional(v.string()),
    preferences: v.optional(v.any()),
    createdAt: v.number(),
  }).index("by_email", ["email"]),

  // Audio Samples for Raga Detection
  audioSamples: defineTable({
    userId: v.id("users"),
    ragaId: v.id("ragas"),
    audioFileId: v.id("_storage"),
    filename: v.string(),
    duration: v.number(),
    sampleRate: v.number(),
    format: v.string(),
    description: v.optional(v.string()),
    isPublic: v.boolean(),
    createdAt: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_raga", ["ragaId"])
    .index("by_public", ["isPublic"]),

  // Raga Detection Results
  ragaDetections: defineTable({
    userId: v.id("users"),
    audioSampleId: v.id("audioSamples"),
    predictions: v.array(v.object({
      raga: v.string(),
      probability: v.number(),
      info: v.optional(v.any()),
    })),
    confidence: v.number(),
    processingTime: v.number(),
    createdAt: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_audio_sample", ["audioSampleId"])
    .index("by_created_at", ["createdAt"]),
});
