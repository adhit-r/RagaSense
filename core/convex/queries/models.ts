import { query } from "./_generated/server";
import { v } from "convex/values";

// Get information about available models
export const getAvailableModels = query({
  args: {},
  handler: async (ctx) => {
    return [
      {
        id: "ragasense",
        name: "RagaSense Internal",
        description: "Our internal model with 106 advanced audio features",
        capabilities: {
          traditions: ["Carnatic", "Hindustani"],
          features: 106,
          accuracy: "85%+",
          tonicDetection: true,
          traditionClassification: true,
          maxRagas: 1402,
          featureTypes: [
            "MFCC", "Spectral", "Chroma", "Tonnetz", 
            "Rhythm", "Energy", "Contrast", "Mel",
            "Gamaka", "Meend", "Shruti", "Cultural"
          ],
        },
        status: "available",
        version: "1.0.0",
        lastUpdated: "2024-09-05",
      },
      {
        id: "carnaticClassifier",
        name: "Carnatic Raga Classifier",
        description: "CNN trained on 10,000+ hours of Carnatic audio from YouTube",
        capabilities: {
          traditions: ["Carnatic"],
          features: "CNN-based",
          accuracy: "High",
          tonicDetection: false,
          traditionClassification: false,
          maxRagas: 150,
          trainingData: "10,000+ hours",
          modelType: "Convolutional Neural Network",
        },
        status: "available",
        version: "1.0.0",
        lastUpdated: "2024-01-01",
        reference: "https://huggingface.co/spaces/sanjeevraja/CarnaticRagaClassifier",
        paper: "YouTube-based Carnatic Raga Classification",
      },
      {
        id: "ragaDetector",
        name: "RagaDetector (Reference)",
        description: "Sequential Pitch Distribution approach for raga detection",
        capabilities: {
          traditions: ["Carnatic", "Hindustani"],
          features: "Sequential Pitch Distribution",
          accuracy: "Research-grade",
          tonicDetection: true,
          traditionClassification: true,
          maxRagas: "Research dataset",
          approach: "Sequential Pitch Distributions",
        },
        status: "reference",
        version: "1.0.0",
        lastUpdated: "2023-01-01",
        paper: "https://aimc2023.pubpub.org/pub/j9v30p0j",
        github: "https://github.com/VishwaasHegde/RagaDetector",
        authors: "Narasinh, V., & Raja, S. (2023)",
      },
    ];
  },
});

// Get detailed information about a specific model
export const getModelInfo = query({
  args: {
    modelId: v.string(),
  },
  handler: async (ctx, args) => {
    const models = await ctx.runQuery("models:getAvailableModels", {});
    const model = models.find((m: any) => m.id === args.modelId);
    
    if (!model) {
      throw new Error(`Model ${args.modelId} not found`);
    }

    // Add additional details based on model
    switch (args.modelId) {
      case "ragasense":
        return {
          ...model,
          technicalDetails: {
            architecture: "Multi-stream Neural Network",
            featureExtraction: "Advanced audio feature engineering",
            trainingData: "1,402 unique ragas (603 Carnatic + 799 Hindustani)",
            culturalFeatures: [
              "Gamaka detection (Carnatic)",
              "Meend tracking (Hindustani)",
              "Shruti complexity analysis",
              "Performance structure analysis",
              "Ornamentation density measurement"
            ],
            performance: {
              traditionAccuracy: "95%+",
              ragaAccuracy: "85%+",
              tonicMAE: "≤15 cents",
              processingTime: "~150ms"
            }
          }
        };
      
      case "carnaticClassifier":
        return {
          ...model,
          technicalDetails: {
            architecture: "Convolutional Neural Network",
            featureExtraction: "Mel-spectrogram based",
            trainingData: "YouTube collection of Carnatic music",
            dataset: "https://ramanarunachalam.github.io/Music/Carnatic/carnatic.html",
            performance: {
              accuracy: "High on YouTube data",
              processingTime: "~200ms",
              inputRequirement: "30+ seconds recommended"
            }
          }
        };
      
      case "ragaDetector":
        return {
          ...model,
          technicalDetails: {
            architecture: "Sequential Pitch Distribution",
            featureExtraction: "CREPE-based pitch tracking",
            approach: "Multi-pitch histogram with weighted approach",
            performance: {
              accuracy: "Research-grade",
              tonicDetection: "Advanced YIN algorithm",
              processingTime: "~300ms"
            }
          }
        };
      
      default:
        return model;
    }
  },
});

// Get model comparison
export const getModelComparison = query({
  args: {},
  handler: async (ctx) => {
    const models = await ctx.runQuery("models:getAvailableModels", {});
    
    return {
      summary: {
        totalModels: models.length,
        availableModels: models.filter((m: any) => m.status === "available").length,
        referenceModels: models.filter((m: any) => m.status === "reference").length,
      },
      comparison: models.map((model: any) => ({
        id: model.id,
        name: model.name,
        traditions: model.capabilities.traditions,
        tonicDetection: model.capabilities.tonicDetection,
        traditionClassification: model.capabilities.traditionClassification,
        maxRagas: model.capabilities.maxRagas,
        status: model.status,
      })),
      recommendations: {
        bestForCarnatic: "carnaticClassifier",
        bestForHindustani: "ragasense",
        bestForBoth: "ragasense",
        bestForResearch: "ragaDetector",
        mostAccurate: "ragasense",
        fastest: "ragasense",
      }
    };
  },
});
