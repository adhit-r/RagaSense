import { action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";

// Cloud Run ML API integration
export const detectRaga = action({
  args: { audioFileId: v.id("_storage") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    // Get audio file URL from Convex Storage
    const audioUrl = await ctx.storage.getUrl(args.audioFileId);
    
    // Call Google Cloud Run ML API
    const response = await fetch(process.env.CLOUD_RUN_ML_URL + "/detect", {
      method: "POST",
      headers: {
        "Content-Type": "multipart/form-data",
      },
      body: JSON.stringify({
        audio_url: audioUrl,
      }),
    });

    if (!response.ok) {
      throw new Error("ML API call failed");
    }

    const result = await response.json();
    
    // Store detection result in Convex
    const detectionId = await ctx.runMutation(api.ragaDetections.create, {
      userId: identity.subject,
      audioSampleId: args.audioFileId,
      predictions: result.predictions,
      confidence: result.predictions[0]?.confidence || 0,
      processingTime: result.processing_time,
    });

    return detectionId;
  },
});

// Get ML API health status
export const getMLAPIHealth = action({
  args: {},
  handler: async (ctx) => {
    try {
      const response = await fetch(process.env.CLOUD_RUN_ML_URL + "/health");
      return await response.json();
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        service: "raga-detection-api"
      };
    }
  },
});

// Get model status
export const getModelStatus = action({
  args: {},
  handler: async (ctx) => {
    try {
      const response = await fetch(process.env.CLOUD_RUN_ML_URL + "/models/status");
      return await response.json();
    } catch (error) {
      return {
        raga_classifier_loaded: false,
        feature_extractor_loaded: false,
        error: error.message
      };
    }
  },
});
