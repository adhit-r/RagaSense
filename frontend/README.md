# Frontend (React)

This directory contains the React frontend for the Raga Detector project. It provides a web interface for uploading audio, viewing raga predictions, and exploring raga data.

## Features
- Upload audio and get raga predictions
- View raga details and analysis
- Compare ragas
- Responsive UI

## How to Run

1. **Install dependencies (if not already):**
   ```sh
   npm install
   ```
2. **Start the development server:**
   ```sh
   npm start
   ```
   The app will run at [http://localhost:3000](http://localhost:3000)

## Backend Connection
- The frontend expects the backend API at `http://localhost:8000`.
- You can change the API URL in the frontend config or `.env` if needed.

## Main Files
- `src/App.js` — Main app logic
- `src/pages/` — Page components
- `src/components/` — UI components

## More
- For backend/API, see [../app/README.md](../app/README.md)
- For ML, see [../app/ml/README.md](../app/ml/README.md) 