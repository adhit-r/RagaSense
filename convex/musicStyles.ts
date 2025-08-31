
// convex/musicStyles.ts
import { v } from "convex/values";
import { query } from "./_generated/server";

export const getMusicStyles = query({
  args: {},
  handler: async (ctx, args) => {
    return [
      {
        id: "alapana",
        name: "Alapana",
        description: "Slow, meditative exploration of the raga",
        duration: "2-5 minutes",
        complexity: "beginner"
      },
      {
        id: "kriti",
        name: "Kriti",
        description: "Traditional composition with lyrics",
        duration: "5-10 minutes", 
        complexity: "intermediate"
      },
      {
        id: "varnam",
        name: "Varnam",
        description: "Energetic composition with complex patterns",
        duration: "3-7 minutes",
        complexity: "advanced"
      },
      {
        id: "thillana",
        name: "Thillana", 
        description: "Rhythmic composition with percussion",
        duration: "2-4 minutes",
        complexity: "intermediate"
      }
    ];
  },
});
