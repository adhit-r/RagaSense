# Convex Migration Plan for Raga Detector

## Overview
Migration from PostgreSQL + SQLAlchemy to Convex for better real-time capabilities, built-in auth, and simplified development.

## Why Convex for This Project?

### 🎵 **Perfect for Music Generation**
- **Real-time Progress**: Live updates during AI music generation
- **File Storage**: Built-in storage for generated audio files
- **User Sessions**: Track generation history and favorites
- **Collaboration**: Share and collaborate on generated music
- **Offline Support**: Works offline with automatic sync

### 🚀 **Technical Benefits**
- **TypeScript First**: Excellent TypeScript support
- **Built-in Auth**: Comprehensive authentication system
- **Edge Functions**: Serverless functions for AI processing
- **Real-time Queries**: Live updates without WebSocket setup
- **Scalable**: Handles complex queries efficiently

## Migration Strategy

### Phase 1: Setup and Configuration
1. **Install Convex**
2. **Configure TypeScript**
3. **Set up authentication**
4. **Create basic schema**

### Phase 2: Data Migration
1. **Export PostgreSQL data**
2. **Transform data for Convex**
3. **Import data to Convex**
4. **Verify data integrity**

### Phase 3: API Migration
1. **Replace SQLAlchemy with Convex queries**
2. **Update authentication**
3. **Implement real-time features**
4. **Add file storage**

### Phase 4: Frontend Integration
1. **Update API calls**
2. **Add real-time subscriptions**
3. **Implement offline support**
4. **Add file upload/download**

## Current vs Convex Architecture

### Current (PostgreSQL + SQLAlchemy)
```
Frontend (Lynx) → FastAPI → SQLAlchemy → PostgreSQL
```

### Convex Architecture
```
Frontend (Lynx) → Convex Client → Convex Backend → Convex Database
                ↓
            Real-time Queries
                ↓
            File Storage
                ↓
            Authentication
```

## Schema Migration

### Current PostgreSQL Schema
```sql
-- Ragas table
CREATE TABLE ragas (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    tradition VARCHAR(50),
    arohana JSONB,
    avarohana JSONB,
    -- ... many more fields
);

-- Users table (if exists)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    -- ... auth fields
);
```

### Convex Schema
```typescript
// schema.ts
import { defineSchema, defineTable } from "convex/schema";
import { v } from "convex/values";

export default defineSchema({
  // Ragas table
  ragas: defineTable({
    name: v.string(),
    tradition: v.union(v.literal("Hindustani"), v.literal("Carnatic")),
    arohana: v.array(v.string()),
    avarohana: v.array(v.string()),
    vadi: v.optional(v.string()),
    samvadi: v.optional(v.string()),
    time: v.optional(v.array(v.string())),
    season: v.optional(v.array(v.string())),
    rasa: v.optional(v.array(v.string())),
    mood: v.optional(v.array(v.string())),
    description: v.optional(v.string()),
    audioFeatures: v.optional(v.any()),
    // ... other fields
  }).index("by_name", ["name"]).index("by_tradition", ["tradition"]),

  // Music Generation History
  musicGenerations: defineTable({
    userId: v.id("users"),
    request: v.object({
      musicType: v.union(v.literal("instrumental"), v.literal("vocal")),
      instruments: v.optional(v.any()),
      voice: v.optional(v.any()),
      mood: v.any(),
      theme: v.any(),
      duration: v.number(),
      tempo: v.optional(v.number()),
      key: v.optional(v.string()),
    }),
    status: v.union(
      v.literal("processing"),
      v.literal("completed"),
      v.literal("failed")
    ),
    audioFileId: v.optional(v.id("_storage")),
    metadata: v.optional(v.any()),
    progress: v.optional(v.number()),
    error: v.optional(v.string()),
    isFavorite: v.boolean(),
  })
    .index("by_user", ["userId"])
    .index("by_status", ["status"])
    .index("by_user_favorites", ["userId", "isFavorite"]),

  // Users (handled by Convex Auth)
  users: defineTable({
    email: v.string(),
    name: v.optional(v.string()),
    preferences: v.optional(v.any()),
    createdAt: v.number(),
  }).index("by_email", ["email"]),
});
```

## Implementation Plan

### 1. Install Convex
```bash
# In frontend directory
bun add convex
bunx convex dev
```

### 2. Configure Convex
```typescript
// convex/convex.json
{
  "project": "raga-detector",
  "team": "your-team",
  "prodUrl": "https://your-project.convex.cloud",
  "functions": {
    "codegen": {
      "enabled": true
    }
  },
  "auth": {
    "enabled": true,
    "domain": "your-domain.com",
    "applicationID": "your-auth0-id"
  }
}
```

### 3. Create Queries and Mutations
```typescript
// convex/ragas.ts
import { query, mutation } from "./_generated/server";
import { v } from "convex/values";

// Get all ragas
export const getAll = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("ragas").collect();
  },
});

// Get raga by name
export const getByName = query({
  args: { name: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("ragas")
      .withIndex("by_name", (q) => q.eq("name", args.name))
      .first();
  },
});

// Search ragas
export const search = query({
  args: { 
    query: v.string(),
    tradition: v.optional(v.union(v.literal("Hindustani"), v.literal("Carnatic")))
  },
  handler: async (ctx, args) => {
    let q = ctx.db.query("ragas");
    
    if (args.tradition) {
      q = q.withIndex("by_tradition", (q) => q.eq("tradition", args.tradition));
    }
    
    const ragas = await q.collect();
    
    // Filter by search query
    return ragas.filter(raga => 
      raga.name.toLowerCase().includes(args.query.toLowerCase()) ||
      raga.description?.toLowerCase().includes(args.query.toLowerCase())
    );
  },
});
```

### 4. Music Generation Functions
```typescript
// convex/musicGeneration.ts
import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Start music generation
export const startGeneration = mutation({
  args: {
    request: v.object({
      musicType: v.union(v.literal("instrumental"), v.literal("vocal")),
      instruments: v.optional(v.any()),
      voice: v.optional(v.any()),
      mood: v.any(),
      theme: v.any(),
      duration: v.number(),
      tempo: v.optional(v.number()),
      key: v.optional(v.string()),
    }),
  },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;

    const generationId = await ctx.db.insert("musicGenerations", {
      userId,
      request: args.request,
      status: "processing",
      progress: 0,
      isFavorite: false,
    });

    // Trigger AI generation (would call your ML service)
    await ctx.scheduler.runAfter(0, "musicGeneration:processGeneration", {
      generationId,
      request: args.request,
    });

    return generationId;
  },
});

// Get user's generation history
export const getUserHistory = query({
  args: { isFavorite: v.optional(v.boolean()) },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const userId = identity.subject;
    let q = ctx.db.query("musicGenerations").withIndex("by_user", (q) => 
      q.eq("userId", userId)
    );

    if (args.isFavorite !== undefined) {
      q = q.withIndex("by_user_favorites", (q) => 
        q.eq("userId", userId).eq("isFavorite", args.isFavorite)
      );
    }

    return await q.collect();
  },
});

// Toggle favorite
export const toggleFavorite = mutation({
  args: { generationId: v.id("musicGenerations") },
  handler: async (ctx, args) => {
    const identity = await ctx.auth.getUserIdentity();
    if (!identity) {
      throw new Error("Not authenticated");
    }

    const generation = await ctx.db.get(args.generationId);
    if (!generation || generation.userId !== identity.subject) {
      throw new Error("Not found or not authorized");
    }

    await ctx.db.patch(args.generationId, {
      isFavorite: !generation.isFavorite,
    });

    return !generation.isFavorite;
  },
});
```

### 5. File Storage
```typescript
// convex/files.ts
import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Generate upload URL
export const generateUploadUrl = mutation({
  args: {},
  handler: async (ctx) => {
    return await ctx.storage.generateUploadUrl();
  },
});

// Get file URL
export const getFileUrl = query({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, args) => {
    return await ctx.storage.getUrl(args.storageId);
  },
});
```

### 6. Frontend Integration
```typescript
// frontend/src/lib/convex.ts
import { ConvexProvider, ConvexReactClient } from "convex/react";

const convex = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL!);

export { convex, ConvexProvider };
```

```typescript
// frontend/src/hooks/useMusicGeneration.ts
import { useMutation, useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";

export function useMusicGeneration() {
  const startGeneration = useMutation(api.musicGeneration.startGeneration);
  const history = useQuery(api.musicGeneration.getUserHistory);
  const toggleFavorite = useMutation(api.musicGeneration.toggleFavorite);

  return {
    startGeneration,
    history,
    toggleFavorite,
  };
}
```

## Benefits After Migration

### 🎵 **Music Generation Features**
- **Real-time Progress**: Live updates during generation
- **File Management**: Automatic audio file storage
- **User History**: Persistent generation history
- **Favorites**: Save and organize generated music
- **Sharing**: Easy sharing of generated music

### 🔐 **Authentication**
- **Built-in Auth**: No need for separate auth service
- **Social Login**: Google, GitHub, etc.
- **User Profiles**: Custom user profiles
- **Permissions**: Fine-grained access control

### ⚡ **Performance**
- **Real-time Queries**: Live updates without polling
- **Edge Functions**: Global deployment
- **Caching**: Automatic query caching
- **Optimistic Updates**: Instant UI updates

### 🛠️ **Development**
- **TypeScript**: Full type safety
- **Hot Reload**: Instant function updates
- **Dashboard**: Built-in admin dashboard
- **Logs**: Real-time function logs

## Migration Timeline

### Week 1: Setup
- Install and configure Convex
- Set up authentication
- Create basic schema

### Week 2: Data Migration
- Export PostgreSQL data
- Transform and import to Convex
- Verify data integrity

### Week 3: API Migration
- Replace SQLAlchemy queries
- Implement real-time features
- Add file storage

### Week 4: Frontend Integration
- Update API calls
- Add real-time subscriptions
- Test and optimize

## Cost Comparison

### Current (PostgreSQL)
- **Database**: $20-50/month (managed PostgreSQL)
- **Auth**: $0-50/month (Auth0 or similar)
- **File Storage**: $5-20/month (S3 or similar)
- **Total**: $25-120/month

### Convex
- **Free Tier**: 1M function calls, 1GB storage
- **Pro Tier**: $25/month for 10M function calls
- **Total**: $0-25/month (likely free for your usage)

## Recommendation

**Yes, migrate to Convex!** It's perfect for your use case because:

1. **Real-time Music Generation**: Live progress updates
2. **Built-in Auth**: No separate auth service needed
3. **File Storage**: Perfect for audio files
4. **TypeScript**: Excellent developer experience
5. **Cost Effective**: Likely free for your usage
6. **Simplified Architecture**: One service instead of multiple

Would you like me to start implementing the Convex migration?
