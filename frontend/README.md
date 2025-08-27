# RagaSense Frontend - Lynx Framework

A modern, cross-platform frontend for RagaSense built with the Lynx framework, featuring Sazhaam-like UX and comprehensive Convex database integration.

## Features

- **Cross-Platform**: Single codebase for Web, iOS, and Android
- **Sazhaam-like UX**: Modern, intuitive user interface
- **Real-time Database**: Full Convex integration with authentication
- **Raga Detection**: Upload and analyze audio files
- **Music Generation**: AI-powered music creation
- **User Management**: Complete user profiles and settings
- **Analytics**: Comprehensive usage tracking
- **File Management**: Audio file storage and organization

## Technology Stack

- **Framework**: Lynx + ReactLynx
- **Build Tool**: Rspeedy
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Database**: Convex (real-time, serverless)
- **Authentication**: Convex Auth
- **State Management**: Convex queries and mutations

## Prerequisites

- Node.js 18 or later
- Bun package manager
- Lynx Explorer (for testing)
- Convex account and project

## Getting Started

### 1. Install Dependencies

```bash
cd frontend
bun install
```

### 2. Set up Convex

1. **Install Convex CLI**:
   ```bash
   bun add -g convex
   ```

2. **Login to Convex**:
   ```bash
   convex login
   ```

3. **Initialize Convex** (if not already done):
   ```bash
   convex dev --configure
   ```

4. **Deploy your schema**:
   ```bash
   convex deploy
   ```

5. **Get your deployment URL** and add it to your environment variables.

### 3. Environment Setup

Copy the example environment file and configure it:

```bash
cp env.example .env.local
```

Edit `.env.local` with your configuration:

```env
# Convex Configuration
NEXT_PUBLIC_CONVEX_URL=your_convex_deployment_url_here

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000

# Google Cloud (for ML services)
NEXT_PUBLIC_GOOGLE_CLOUD_PROJECT=your_project_id_here
NEXT_PUBLIC_GOOGLE_CLOUD_REGION=us-central1
```

### 4. Development

Start the development server:

```bash
bun run dev
```

### 5. Testing with Lynx Explorer

1. Install Lynx Explorer from the [official guide](http://lynxjs.org/guide/start/quick-start.html)
2. Scan the QR code or copy the bundle URL
3. Test the app on your device

## Available Scripts

- `bun run dev` - Start development server
- `bun run build` - Build for all platforms
- `bun run build:web` - Build for web only
- `bun run build:ios` - Build for iOS
- `bun run build:android` - Build for Android
- `bun run preview` - Preview the build
- `bun run lint` - Run linting
- `bun run type-check` - Run TypeScript checks
- `bun run convex:dev` - Start Convex development server
- `bun run convex:deploy` - Deploy Convex functions
- `bun run convex:codegen` - Generate Convex types

## Project Structure

```
frontend/
├── convex/                 # Convex database and functions
│   ├── schema.ts          # Database schema
│   ├── functions/         # Convex functions
│   │   ├── auth.ts        # Authentication
│   │   ├── ragas.ts       # Raga management
│   │   ├── ragaDetection.ts # Detection history
│   │   ├── files.ts       # File management
│   │   ├── musicGeneration.ts # Music generation
│   │   └── analytics.ts   # Analytics tracking
│   └── convex.json        # Convex configuration
├── src/
│   ├── components/        # ReactLynx components
│   ├── hooks/            # Custom hooks for Convex
│   ├── lib/              # Utilities and configurations
│   ├── styles/           # Global styles
│   └── types/            # TypeScript type definitions
├── rspeedy.config.ts     # Lynx build configuration
└── package.json          # Dependencies and scripts
```

## Convex Database Schema

The application uses a comprehensive Convex schema with the following tables:

- **users**: User profiles and authentication
- **ragas**: Raga information and metadata
- **ragaDetections**: Detection history and results
- **files**: Audio file storage and management
- **musicGenerations**: AI music generation requests
- **userSettings**: User preferences and settings
- **userFavorites**: User's favorite ragas
- **analytics**: Usage analytics and tracking

## Authentication

The app uses Convex's built-in authentication system with support for:

- Email/password authentication
- OAuth providers (Google, GitHub, etc.)
- User profile management
- Settings and preferences

## Key Features

### Raga Detection
- Upload audio files (WAV, MP3, OGG, FLAC, M4A)
- Real-time raga detection
- Detection history and statistics
- Export detection results

### Music Generation
- AI-powered music generation
- Raga-specific composition
- Generation history and management
- Background processing support

### User Management
- Complete user profiles
- Settings and preferences
- Favorites and collections
- Usage analytics

### File Management
- Audio file upload and storage
- File organization and search
- File statistics and analytics

## Development Workflow

1. **Database Changes**: Modify `convex/schema.ts` and run `convex deploy`
2. **Function Development**: Edit functions in `convex/functions/`
3. **Frontend Development**: Use ReactLynx components in `src/components/`
4. **Testing**: Use Lynx Explorer for cross-platform testing

## Deployment

### Web Deployment
```bash
bun run build:web
# Deploy the dist/web folder to your hosting provider
```

### Mobile Deployment
```bash
bun run build:ios    # For iOS
bun run build:android # For Android
```

### Convex Deployment
```bash
bun run convex:deploy
```

## Resources

- [Lynx Documentation](http://lynxjs.org/)
- [Convex Documentation](https://docs.convex.dev/)
- [ReactLynx Guide](http://lynxjs.org/guide/react-lynx/)
- [Rspeedy Build Tool](http://lynxjs.org/guide/build-tools/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Lynx Explorer
5. Submit a pull request

## Support

For issues and questions:
- Check the [docs](docs/) folder
- Create an issue on GitHub
- Join our community discussions

---

Built with ❤️ for Indian classical music enthusiasts
