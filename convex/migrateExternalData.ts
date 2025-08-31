// convex/migrateExternalData.ts
import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// Types for our enhanced raga data structure
interface RagaMetadata {
  name: string;
  tradition: "Carnatic" | "Hindustani";
  arohanam?: string;
  avarohanam?: string;
  vadi?: string;
  samvadi?: string;
  characteristicPhrases?: string[];
  compositions?: Composition[];
  audioExamples?: AudioExample[];
  difficulty?: "Beginner" | "Intermediate" | "Advanced";
  mood?: string[];
  timeOfDay?: string[];
  season?: string[];
}

interface Composition {
  name: string;
  composer?: string;
  language?: string;
  type?: string;
  audioUrl?: string;
}

interface AudioExample {
  title: string;
  artist?: string;
  duration: number;
  audioUrl: string;
  quality: "Low" | "Medium" | "High";
}

interface UserProfile {
  userId: string;
  preferences: {
    preferredTradition: "Carnatic" | "Hindustani" | "Both";
    skillLevel: "Beginner" | "Intermediate" | "Advanced";
    favoriteRagas: string[];
  };
  learningProgress: {
    ragasLearned: string[];
    practiceSessions: PracticeSession[];
    accuracyHistory: AccuracyRecord[];
  };
  createdAt: string;
  updatedAt: string;
}

interface PracticeSession {
  ragaId: string;
  sessionType: "Detection" | "Ear Training" | "Composition";
  duration: number;
  accuracy: number;
  timestamp: string;
}

interface AccuracyRecord {
  ragaId: string;
  accuracy: number;
  timestamp: string;
  modelUsed: string;
}

// Migration function to load external data
export const migrateExternalData = mutation({
  args: {},
  handler: async (ctx, args) => {
    console.log("Starting external data migration...");
    
    // This would typically read from the external_music_data directory
    // For now, we'll create enhanced raga data structure
    
    const enhancedRagas: RagaMetadata[] = [
      {
        name: "Hamsadhwani",
        tradition: "Carnatic",
        arohanam: "S R2 G3 P D2 S",
        avarohanam: "S D2 P G3 R2 S",
        vadi: "G3",
        samvadi: "D2",
        characteristicPhrases: [
          "G3 R2 S D2 P G3",
          "S R2 G3 P D2 S",
          "D2 P G3 R2 S"
        ],
        difficulty: "Beginner",
        mood: ["Joyful", "Devotional"],
        timeOfDay: ["Evening"],
        season: ["All Seasons"],
        compositions: [
          {
            name: "Vatapi Ganapatim",
            composer: "Muthuswami Dikshitar",
            language: "Sanskrit",
            type: "Varnam"
          }
        ],
        audioExamples: [
          {
            title: "Hamsadhwani Alapana",
            duration: 180,
            audioUrl: "/audio/hamsadhwani_alapana.mp3",
            quality: "High"
          }
        ]
      },
      {
        name: "Kamboji",
        tradition: "Carnatic",
        arohanam: "S R2 G3 M1 P D2 S",
        avarohanam: "S D2 P M1 G3 R2 S",
        vadi: "M1",
        samvadi: "S",
        characteristicPhrases: [
          "G3 M1 P D2 S",
          "S R2 G3 M1 P",
          "M1 G3 R2 S"
        ],
        difficulty: "Intermediate",
        mood: ["Devotional", "Serene"],
        timeOfDay: ["Evening"],
        season: ["Monsoon"],
        compositions: [
          {
            name: "O Rangasayee",
            composer: "Tyagaraja",
            language: "Telugu",
            type: "Kriti"
          }
        ]
      },
      {
        name: "Atana",
        tradition: "Carnatic",
        arohanam: "S R2 G3 M1 P D2 N3 S",
        avarohanam: "S N3 D2 P M1 G3 R2 S",
        vadi: "M1",
        samvadi: "S",
        characteristicPhrases: [
          "G3 M1 P D2 N3 S",
          "S R2 G3 M1 P",
          "N3 D2 P M1"
        ],
        difficulty: "Intermediate",
        mood: ["Majestic", "Devotional"],
        timeOfDay: ["Evening"],
        season: ["All Seasons"]
      },
      {
        name: "Yaman",
        tradition: "Hindustani",
        arohanam: "S R G M^ P D N S'",
        avarohanam: "S' N D P M^ G R S",
        vadi: "G",
        samvadi: "N",
        characteristicPhrases: [
          "G M^ P D N S'",
          "S' N D P M^ G",
          "M^ G R S"
        ],
        difficulty: "Beginner",
        mood: ["Romantic", "Devotional"],
        timeOfDay: ["Evening"],
        season: ["All Seasons"]
      },
      {
        name: "Bhairavi",
        tradition: "Hindustani",
        arohanam: "S r g M P d N S'",
        avarohanam: "S' N d P M g r S",
        vadi: "M",
        samvadi: "S",
        characteristicPhrases: [
          "g M P d N S'",
          "S' N d P M g",
          "M g r S"
        ],
        difficulty: "Advanced",
        mood: ["Serious", "Devotional"],
        timeOfDay: ["Morning"],
        season: ["Winter"]
      }
    ];

    // Insert enhanced raga data
    const ragaIds = [];
    for (const raga of enhancedRagas) {
      const ragaId = await ctx.db.insert("ragas", {
        name: raga.name,
        tradition: raga.tradition,
        arohanam: raga.arohanam,
        avarohanam: raga.avarohanam,
        vadi: raga.vadi,
        samvadi: raga.samvadi,
        characteristicPhrases: raga.characteristicPhrases,
        difficulty: raga.difficulty,
        mood: raga.mood,
        timeOfDay: raga.timeOfDay,
        season: raga.season,
        compositions: raga.compositions,
        audioExamples: raga.audioExamples,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      });
      ragaIds.push(ragaId);
    }

    console.log(`Migrated ${ragaIds.length} ragas with enhanced metadata`);
    return { success: true, ragasMigrated: ragaIds.length };
  },
});

// Function to get enhanced raga data
export const getEnhancedRagas = query({
  args: {
    tradition: v.optional(v.string()),
    difficulty: v.optional(v.string()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    let query = ctx.db.query("ragas");
    
    if (args.tradition) {
      query = query.filter((q) => q.eq(q.field("tradition"), args.tradition));
    }
    
    if (args.difficulty) {
      query = query.filter((q) => q.eq(q.field("difficulty"), args.difficulty));
    }
    
    if (args.limit) {
      query = query.take(args.limit);
    }
    
    return await query.collect();
  },
});

// Function to get raga by name with full metadata
export const getRagaByName = query({
  args: { name: v.string() },
  handler: async (ctx, args) => {
    const raga = await ctx.db
      .query("ragas")
      .filter((q) => q.eq(q.field("name"), args.name))
      .first();
    
    return raga;
  },
});

// Function to create user profile
export const createUserProfile = mutation({
  args: {
    userId: v.string(),
    preferences: v.object({
      preferredTradition: v.string(),
      skillLevel: v.string(),
      favoriteRagas: v.array(v.string()),
    }),
  },
  handler: async (ctx, args) => {
    const profileId = await ctx.db.insert("userProfiles", {
      userId: args.userId,
      preferences: args.preferences,
      learningProgress: {
        ragasLearned: [],
        practiceSessions: [],
        accuracyHistory: []
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    });
    
    return profileId;
  },
});

// Function to record practice session
export const recordPracticeSession = mutation({
  args: {
    userId: v.string(),
    ragaId: v.id("ragas"),
    sessionType: v.string(),
    duration: v.number(),
    accuracy: v.number(),
    modelUsed: v.string(),
  },
  handler: async (ctx, args) => {
    // Get user profile
    const profile = await ctx.db
      .query("userProfiles")
      .filter((q) => q.eq(q.field("userId"), args.userId))
      .first();
    
    if (!profile) {
      throw new Error("User profile not found");
    }
    
    const practiceSession: PracticeSession = {
      ragaId: args.ragaId,
      sessionType: args.sessionType as "Detection" | "Ear Training" | "Composition",
      duration: args.duration,
      accuracy: args.accuracy,
      timestamp: new Date().toISOString()
    };
    
    const accuracyRecord: AccuracyRecord = {
      ragaId: args.ragaId,
      accuracy: args.accuracy,
      timestamp: new Date().toISOString(),
      modelUsed: args.modelUsed
    };
    
    // Update profile with new session and accuracy record
    await ctx.db.patch(profile._id, {
      learningProgress: {
        ragasLearned: profile.learningProgress.ragasLearned,
        practiceSessions: [...profile.learningProgress.practiceSessions, practiceSession],
        accuracyHistory: [...profile.learningProgress.accuracyHistory, accuracyRecord]
      },
      updatedAt: new Date().toISOString()
    });
    
    return { success: true };
  },
});

// Function to get user learning progress
export const getUserProgress = query({
  args: { userId: v.string() },
  handler: async (ctx, args) => {
    const profile = await ctx.db
      .query("userProfiles")
      .filter((q) => q.eq(q.field("userId"), args.userId))
      .first();
    
    return profile?.learningProgress;
  },
});

// Function to get ragas by difficulty for learning
export const getRagasByDifficulty = query({
  args: {
    difficulty: v.string(),
    tradition: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    let query = ctx.db.query("ragas")
      .filter((q) => q.eq(q.field("difficulty"), args.difficulty));
    
    if (args.tradition) {
      query = query.filter((q) => q.eq(q.field("tradition"), args.tradition));
    }
    
    return await query.collect();
  },
});

// Function to search ragas with advanced filters
export const searchRagasAdvanced = query({
  args: {
    searchTerm: v.optional(v.string()),
    tradition: v.optional(v.string()),
    difficulty: v.optional(v.string()),
    mood: v.optional(v.string()),
    timeOfDay: v.optional(v.string()),
    season: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    let query = ctx.db.query("ragas");
    
    if (args.searchTerm) {
      query = query.filter((q) => 
        q.or(
          q.gte(q.field("name"), args.searchTerm),
          q.lt(q.field("name"), args.searchTerm + "\uffff")
        )
      );
    }
    
    if (args.tradition) {
      query = query.filter((q) => q.eq(q.field("tradition"), args.tradition));
    }
    
    if (args.difficulty) {
      query = query.filter((q) => q.eq(q.field("difficulty"), args.difficulty));
    }
    
    if (args.mood) {
      query = query.filter((q) => q.includes(q.field("mood"), args.mood));
    }
    
    if (args.timeOfDay) {
      query = query.filter((q) => q.includes(q.field("timeOfDay"), args.timeOfDay));
    }
    
    if (args.season) {
      query = query.filter((q) => q.includes(q.field("season"), args.season));
    }
    
    return await query.collect();
  },
});

