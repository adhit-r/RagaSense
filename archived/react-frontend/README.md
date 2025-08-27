# Archived React Frontend

This directory contains the original React frontend that has been archived as part of the migration to Lynx.

## Original Structure
- **Framework**: React 18 with TypeScript
- **UI Library**: Radix UI components with Tailwind CSS
- **State Management**: React Query for API state
- **Audio Handling**: WaveSurfer.js for audio visualization
- **Build Tool**: Vite

## Migration Notes
- Archived on: August 27, 2024
- Reason: Migration to Lynx framework for better performance and developer experience
- Status: All functionality preserved in Lynx implementation

## Key Features (Preserved in Lynx)
- Drag-and-drop audio upload
- Real-time audio playback and visualization
- Interactive raga analysis
- Responsive design with dark/light themes
- API integration with FastAPI backend

## To Restore (if needed)
```bash
# Copy back to frontend directory
cp -r archived/react-frontend/* frontend/

# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

## Lynx Migration Benefits
- Better performance with native compilation
- Improved developer experience
- Enhanced type safety
- Modern framework features
- Better integration with backend APIs 