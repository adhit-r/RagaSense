import { action } from "./_generated/server";
import { v } from "convex/values";

// Store audio analysis results in Convex
export const storeAnalysis = action({
  args: {
    audio_file: v.string(),
    analysis_results: v.any(),
    timestamp: v.string(),
    status: v.string(),
  },
  handler: async (ctx, args) => {
    try {
      // Store the analysis results
      const result = await ctx.runMutation(async (ctx) => {
        return await ctx.db.insert("audioAnalysis", {
          audio_file: args.audio_file,
          analysis_results: args.analysis_results,
          timestamp: args.timestamp,
          status: args.status,
          created_at: new Date().toISOString(),
        });
      });

      return {
        success: true,
        id: result,
        message: "Audio analysis stored successfully",
      };
    } catch (error) {
      console.error("Error storing audio analysis:", error);
      return {
        success: false,
        error: "Failed to store audio analysis",
      };
    }
  },
});

// Retrieve audio analysis results from Convex
export const getAnalysis = action({
  args: {
    audio_file: v.string(),
  },
  handler: async (ctx, args) => {
    try {
      // Retrieve the analysis results
      const result = await ctx.runQuery(async (ctx) => {
        return await ctx.db
          .query("audioAnalysis")
          .filter((q) => q.eq(q.field("audio_file"), args.audio_file))
          .order("desc")
          .first();
      });

      if (result) {
        return {
          success: true,
          data: result,
        };
      } else {
        return {
          success: false,
          error: "Analysis not found",
        };
      }
    } catch (error) {
      console.error("Error retrieving audio analysis:", error);
      return {
        success: false,
        error: "Failed to retrieve audio analysis",
      };
    }
  },
});

// Get all audio analysis results
export const getAllAnalysis = action({
  args: {},
  handler: async (ctx) => {
    try {
      const results = await ctx.runQuery(async (ctx) => {
        return await ctx.db
          .query("audioAnalysis")
          .order("desc")
          .collect();
      });

      return {
        success: true,
        data: results,
        count: results.length,
      };
    } catch (error) {
      console.error("Error retrieving all audio analysis:", error);
      return {
        success: false,
        error: "Failed to retrieve audio analysis",
      };
    }
  },
});

// Get analysis statistics
export const getAnalysisStats = action({
  args: {},
  handler: async (ctx) => {
    try {
      const results = await ctx.runQuery(async (ctx) => {
        return await ctx.db
          .query("audioAnalysis")
          .order("desc")
          .collect();
      });

      // Calculate statistics
      const total = results.length;
      const correct_predictions = results.filter(
        (r) => r.analysis_results?.correct_prediction
      ).length;
      const accuracy = total > 0 ? correct_predictions / total : 0;

      // Calculate average confidence
      const confidences = results
        .map((r) => r.analysis_results?.confidence)
        .filter((c) => c !== null && c !== undefined);
      const avg_confidence =
        confidences.length > 0
          ? confidences.reduce((a, b) => a + b, 0) / confidences.length
          : 0;

      // Tradition breakdown
      const carnatic_files = results.filter(
        (r) => r.analysis_results?.expected_tradition === "Carnatic"
      ).length;
      const hindustani_files = results.filter(
        (r) => r.analysis_results?.expected_tradition === "Hindustani"
      ).length;

      return {
        success: true,
        stats: {
          total_files: total,
          correct_predictions,
          accuracy: accuracy * 100, // Convert to percentage
          average_confidence: avg_confidence * 100, // Convert to percentage
          tradition_breakdown: {
            carnatic: carnatic_files,
            hindustani: hindustani_files,
          },
        },
      };
    } catch (error) {
      console.error("Error calculating analysis statistics:", error);
      return {
        success: false,
        error: "Failed to calculate statistics",
      };
    }
  },
});

// Delete audio analysis
export const deleteAnalysis = action({
  args: {
    id: v.id("audioAnalysis"),
  },
  handler: async (ctx, args) => {
    try {
      await ctx.runMutation(async (ctx) => {
        await ctx.db.delete(args.id);
      });

      return {
        success: true,
        message: "Audio analysis deleted successfully",
      };
    } catch (error) {
      console.error("Error deleting audio analysis:", error);
      return {
        success: false,
        error: "Failed to delete audio analysis",
      };
    }
  },
});
