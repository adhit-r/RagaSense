import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // Files table for storing uploaded audio files
  files: defineTable({
    userId: v.string(),
    filename: v.string(),
    storageId: v.string(),
    contentType: v.string(),
    size: v.number(),
    uploadedAt: v.string(),
  }).index("by_user", ["userId"]),

  // Ragas table for storing raga information
  ragas: defineTable({
    name: v.string(),
    tradition: v.string(), // "Carnatic" or "Hindustani"
    parentScale: v.string(),
    description: v.string(),
    characteristics: v.array(v.string()),
    popularCompositions: v.array(v.string()),
    mood: v.string(),
    timeOfDay: v.string(),
    season: v.string(),
  }).index("by_tradition", ["tradition"]),

  // Classifications table for storing raga detection results
  classifications: defineTable({
    userId: v.string(),
    fileId: v.string(),
    ragaName: v.string(),
    confidence: v.number(),
    tradition: v.string(),
    features: v.array(v.number()),
    timestamp: v.string(),
  }).index("by_user", ["userId"]),

  // Music Generations table for storing generated music
  musicGenerations: defineTable({
    userId: v.string(),
    raga: v.string(),
    style: v.string(),
    duration: v.number(),
    status: v.string(),
    resultUrl: v.optional(v.string()),
    timestamp: v.string(),
  }).index("by_user", ["userId"]),

  // User Profiles table for storing user information
  userProfiles: defineTable({
    userId: v.string(),
    name: v.string(),
    email: v.string(),
    preferences: v.any(),
    createdAt: v.string(),
    updatedAt: v.string(),
  }).index("by_user", ["userId"]),

  // Audio Analysis Results table for storing real dataset analysis
  audioAnalysis: defineTable({
    audio_file: v.string(),
    analysis_results: v.any(),
    timestamp: v.string(),
    status: v.string(),
    created_at: v.string(),
  }).index("by_audio_file", ["audio_file"]),

  // Multi-Model Detections table for storing multi-model raga detection results
  multiModelDetections: defineTable({
    userId: v.string(),
    audioFileId: v.string(),
    results: v.array(v.object({
      modelName: v.string(),
      raga: v.string(),
      confidence: v.number(),
      tonic: v.optional(v.string()),
      tradition: v.optional(v.string()),
      topPredictions: v.optional(v.array(v.object({
        raga: v.string(),
        confidence: v.number(),
        tradition: v.optional(v.string()),
      }))),
      metadata: v.optional(v.any()),
      error: v.optional(v.string()),
    })),
    processingTime: v.string(),
    timestamp: v.string(),
    createdAt: v.string(),
  }).index("by_user", ["userId"]),
});
