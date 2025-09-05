import { action } from "./_generated/server";
import { v } from "convex/values";

// Multi-model raga detection action
export const detectRagaMultiModel = action({
  args: {
    request: v.object({
      audioFileId: v.string(),
      selectedModels: v.array(v.string()),
      tradition: v.optional(v.string()),
      tonic: v.optional(v.string()),
      duration: v.number(),
    }),
    userId: v.string(),
  },
  handler: async (ctx, args) => {
    const startTime = Date.now();
    
    try {
      // Get audio file info
      const audioFile = await ctx.runQuery("files:getFileById", {
        fileId: args.request.audioFileId,
      });

      if (!audioFile) {
        throw new Error("Audio file not found");
      }

      const results = [];

      // Process each selected model
      for (const modelName of args.request.selectedModels) {
        try {
          let result;
          
          switch (modelName) {
            case "ragasense":
              result = await detectWithRagaSense(ctx, audioFile, args.request);
              break;
            case "carnaticClassifier":
              result = await detectWithCarnaticClassifier(ctx, audioFile, args.request);
              break;
            case "ragaDetector":
              result = await detectWithRagaDetector(ctx, audioFile, args.request);
              break;
            default:
              throw new Error(`Unknown model: ${modelName}`);
          }

          results.push(result);
        } catch (error) {
          // Add error result for this model
          results.push({
            modelName: modelName,
            raga: "Unknown",
            confidence: 0,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      const processingTime = `${Date.now() - startTime}ms`;

      // Save detection results
      const detectionId = await ctx.runMutation("classifications:createMultiModelDetection", {
        userId: args.userId,
        audioFileId: args.request.audioFileId,
        results: results,
        processingTime: processingTime,
        timestamp: new Date().toISOString(),
      });

      return {
        audioFileId: args.request.audioFileId,
        results: results,
        processingTime: processingTime,
        timestamp: new Date().toISOString(),
        detectionId: detectionId,
      };
    } catch (error) {
      throw new Error(`Multi-model detection failed: ${error}`);
    }
  },
});

// RagaSense internal model detection
async function detectWithRagaSense(ctx: any, audioFile: any, request: any) {
  // TODO: Implement RagaSense model inference
  // This would call our internal model with 106 features
  
  // For now, return a mock result
  return {
    modelName: "RagaSense Internal",
    raga: "Kalyani",
    confidence: 0.92,
    tonic: request.tonic || "C",
    tradition: request.tradition || "Carnatic",
    topPredictions: [
      { raga: "Kalyani", confidence: 0.92, tradition: "Carnatic" },
      { raga: "Hamsadhwani", confidence: 0.15, tradition: "Carnatic" },
      { raga: "Mohanam", confidence: 0.08, tradition: "Carnatic" },
    ],
    metadata: {
      features: 106,
      processingTime: "150ms",
      modelVersion: "1.0.0",
    },
  };
}

// Carnatic Raga Classifier detection
async function detectWithCarnaticClassifier(ctx: any, audioFile: any, request: any) {
  // TODO: Integrate with the carnatic-raga-classifier
  // This would call the CNN model trained on 10,000+ hours
  
  // For now, return a mock result
  return {
    modelName: "Carnatic Raga Classifier",
    raga: "Kalyani",
    confidence: 0.88,
    tradition: "Carnatic",
    topPredictions: [
      { raga: "Kalyani", confidence: 0.88, tradition: "Carnatic" },
      { raga: "Bhairavi", confidence: 0.12, tradition: "Carnatic" },
      { raga: "Shankarabharanam", confidence: 0.08, tradition: "Carnatic" },
    ],
    metadata: {
      modelType: "CNN",
      trainingData: "10,000+ hours",
      maxRagas: 150,
    },
  };
}

// RagaDetector (Reference) detection
async function detectWithRagaDetector(ctx: any, audioFile: any, request: any) {
  // TODO: Integrate with the RagaDetector reference implementation
  // This would use the Sequential Pitch Distribution approach
  
  // For now, return a mock result
  return {
    modelName: "RagaDetector (Reference)",
    raga: "Kalyani",
    confidence: 0.85,
    tonic: request.tonic || "C",
    tradition: request.tradition || "Carnatic",
    topPredictions: [
      { raga: "Kalyani", confidence: 0.85, tradition: "Carnatic" },
      { raga: "Hamsadhwani", confidence: 0.20, tradition: "Carnatic" },
      { raga: "Mohanam", confidence: 0.10, tradition: "Carnatic" },
    ],
    metadata: {
      approach: "Sequential Pitch Distribution",
      paper: "https://aimc2023.pubpub.org/pub/j9v30p0j",
      tonicDetection: true,
    },
  };
}

// Get model information
export const getModelInfo = action({
  args: {
    modelName: v.string(),
  },
  handler: async (ctx, args) => {
    const modelInfo = {
      ragasense: {
        name: "RagaSense Internal",
        description: "Our internal model with 106 advanced features",
        capabilities: {
          traditions: ["Carnatic", "Hindustani"],
          features: 106,
          accuracy: "85%+",
          tonicDetection: true,
          traditionClassification: true,
          maxRagas: 1402,
        },
        status: "available",
      },
      carnaticClassifier: {
        name: "Carnatic Raga Classifier",
        description: "CNN trained on 10,000+ hours of Carnatic audio",
        capabilities: {
          traditions: ["Carnatic"],
          features: "CNN-based",
          accuracy: "High",
          tonicDetection: false,
          traditionClassification: false,
          maxRagas: 150,
          trainingData: "10,000+ hours",
        },
        status: "available",
        reference: "https://huggingface.co/spaces/sanjeevraja/CarnaticRagaClassifier",
      },
      ragaDetector: {
        name: "RagaDetector (Reference)",
        description: "Sequential Pitch Distribution approach",
        capabilities: {
          traditions: ["Carnatic", "Hindustani"],
          features: "Sequential Pitch Distribution",
          accuracy: "Research-grade",
          tonicDetection: true,
          traditionClassification: true,
          maxRagas: "Research dataset",
        },
        status: "reference",
        paper: "https://aimc2023.pubpub.org/pub/j9v30p0j",
        github: "https://github.com/VishwaasHegde/RagaDetector",
      },
    };

    return modelInfo[args.modelName as keyof typeof modelInfo] || {
      name: args.modelName,
      description: "Unknown model",
      status: "unavailable",
    };
  },
});
