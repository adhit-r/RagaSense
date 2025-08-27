# **Codebase Organization & Structure**

## **Overview**

This document provides a comprehensive guide to the organized codebase structure, making it developer-friendly and easy to navigate. We've consolidated scattered files and created a clear, logical structure.

## **Current Codebase Structure**

```
raga_detector/
├── frontend/                    # Complete Lynx Frontend
│   ├── src/                     # Source code
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/              # Page components
│   │   ├── api/                # API integration
│   │   ├── types/              # TypeScript definitions
│   │   ├── styles/             # CSS and styling
│   │   ├── App.tsx             # Main app component
│   │   └── main.tsx            # Entry point
│   ├── convex/                 # Convex Backend Functions
│   │   ├── _generated/         # Auto-generated Convex types
│   │   ├── schema.ts           # Database schema
│   │   ├── ragas.ts            # Raga operations
│   │   ├── musicGeneration.ts  # AI music generation
│   │   ├── files.ts            # File operations
│   │   └── ml_integration.ts   # ML API integration
│   ├── package.json            # Frontend dependencies
│   ├── lynx.config.ts          # Lynx configuration
│   ├── tailwind.config.js      # Tailwind CSS config
│   └── tsconfig.json           # TypeScript config
├── ml/                         # Machine Learning Components
│   ├── cloud_run_app.py        # FastAPI ML API
│   ├── raga_classifier.py      # Raga classification model
│   ├── data_loader.py          # Data loading utilities
│   └── cloud_run_setup.py      # Google Cloud Run setup
├── scripts/                    # Utility Scripts
│   ├── download_training_data.py # Download ML training data
│   └── upload_models_to_gcs.py # Upload models to Google Cloud
├── docs/                       # Comprehensive Documentation
│   ├── README.md               # Documentation index
│   ├── QUICK_DEPLOYMENT_GUIDE.md # Quick start guide
│   ├── ML_RAGA_DETECTION_SCIENTIFIC.md # ML scientific foundation
│   ├── AI_MUSIC_GENERATION_SCIENTIFIC.md # AI generation scientific foundation
│   ├── GOOGLE_CLOUD_RUN_ML_SETUP.md # ML hosting setup
│   ├── CONVEX_IMPLEMENTATION_COMPLETE.md # Backend implementation
│   ├── ANSWERS_TO_YOUR_QUESTIONS.md # FAQ and troubleshooting
│   ├── API_DOCS.md             # API documentation
│   ├── TODO.md                 # Current tasks and roadmap
│   └── DOCS_STRUCTURE.md       # Documentation organization
├── tests/                      # Test Files
├── archived/                   # Archived Code
│   └── react-frontend/         # Original React frontend
├── backend/                    # Legacy Backend (PostgreSQL)
├── migrations/                 # Database migrations
├── uploads/                    # File uploads
├── venv/                       # Python virtual environment
├── README.md                   # Main project README
├── requirements.txt            # Python dependencies
├── requirements_ml.txt         # ML-specific dependencies
├── docker-compose.yml          # Docker configuration
├── Dockerfile                  # Main Dockerfile
├── deploy_to_cloud_run.sh      # ML deployment script
├── run_dev.sh                  # Development startup script
└── .gitignore                  # Git ignore rules
```

## **Organization Changes Made**

### **Consolidated Frontend Structure**
- **Moved `src/` into `frontend/src/`** - All frontend code now in one place
- **Consolidated `convex/` folders** - Single `frontend/convex/` directory
- **Organized components** - Clear separation of concerns

### **Cleaned Up Root Directory**
- **Removed scattered files** - No more confusion with multiple locations
- **Organized scripts** - All utilities in `scripts/` folder
- **Comprehensive docs** - All documentation in `docs/` folder

### **Developer-Friendly Structure**
- **Clear separation** - Frontend, ML, and backend clearly separated
- **Logical grouping** - Related files grouped together
- **Easy navigation** - Intuitive folder structure

## **Key Directories Explained**

### **`frontend/` - Complete Lynx Application**

```
frontend/
├── src/                    # Lynx source code
│   ├── components/         # Reusable UI components
│   │   ├── Button.tsx     # Button component
│   │   ├── Card.tsx       # Card component
│   │   ├── Navbar.tsx     # Navigation bar
│   │   ├── ThemeProvider.tsx # Theme management
│   │   └── Toaster.tsx    # Toast notifications
│   ├── pages/             # Page components
│   │   ├── Home.tsx       # Home page
│   │   ├── RagaDetector.tsx # Raga detection page
│   │   ├── RagaList.tsx   # Raga browsing page
│   │   └── MusicGenerator.tsx # AI music generation
│   ├── api/               # API integration
│   │   ├── client.ts      # HTTP client
│   │   └── ragas.ts       # Raga API functions
│   ├── types/             # TypeScript definitions
│   │   └── index.ts       # All type definitions
│   ├── styles/            # CSS and styling
│   │   └── globals.css    # Global styles
│   ├── App.tsx            # Main app component
│   └── main.tsx           # Entry point
├── convex/                # Convex backend functions
│   ├── _generated/        # Auto-generated types
│   ├── schema.ts          # Database schema
│   ├── ragas.ts           # Raga operations
│   ├── musicGeneration.ts # AI music generation
│   ├── files.ts           # File operations
│   └── ml_integration.ts  # ML API integration
├── package.json           # Dependencies and scripts
├── lynx.config.ts         # Lynx configuration
├── tailwind.config.js     # Tailwind CSS config
└── tsconfig.json          # TypeScript config
```

### **`ml/` - Machine Learning Components**

```
ml/
├── cloud_run_app.py       # FastAPI ML API for Google Cloud Run
├── raga_classifier.py     # Raga classification model
├── data_loader.py         # Data loading and preprocessing
└── cloud_run_setup.py     # Google Cloud Run setup script
```

### **`scripts/` - Utility Scripts**

```
scripts/
├── download_training_data.py  # Download ML training data and models
└── upload_models_to_gcs.py    # Upload models to Google Cloud Storage
```

### **`docs/` - Comprehensive Documentation**

```
docs/
├── README.md                  # Documentation index
├── QUICK_DEPLOYMENT_GUIDE.md  # Quick start guide
├── ML_RAGA_DETECTION_SCIENTIFIC.md # ML scientific foundation
├── AI_MUSIC_GENERATION_SCIENTIFIC.md # AI generation scientific foundation
├── GOOGLE_CLOUD_RUN_ML_SETUP.md # ML hosting setup
├── CONVEX_IMPLEMENTATION_COMPLETE.md # Backend implementation
├── ANSWERS_TO_YOUR_QUESTIONS.md # FAQ and troubleshooting
├── API_DOCS.md                # API documentation
├── TODO.md                    # Current tasks and roadmap
└── DOCS_STRUCTURE.md          # Documentation organization
```

## **Development Workflow**

### **1. Frontend Development**
```bash
# Navigate to frontend
cd frontend

# Install dependencies
bun install

# Start development server
bun run dev

# Build for production
bun run build
```

### **2. Backend Development (Convex)**
```bash
# Start Convex development server
bunx convex dev

# Deploy to production
bunx convex deploy
```

### **3. ML Development**
```bash
# Install Python dependencies
pip install -r requirements_ml.txt

# Run ML API locally
python ml/cloud_run_app.py

# Deploy to Google Cloud Run
./deploy_to_cloud_run.sh
```

### **4. Full Stack Development**
```bash
# Start all services
./run_dev.sh
```

## **File Naming Conventions**

### **Frontend Files**
- **Components**: PascalCase (e.g., `Button.tsx`, `RagaDetector.tsx`)
- **Pages**: PascalCase (e.g., `Home.tsx`, `MusicGenerator.tsx`)
- **Utilities**: camelCase (e.g., `apiClient.ts`, `useRagas.ts`)
- **Types**: camelCase (e.g., `index.ts` in types folder)

### **Backend Files (Convex)**
- **Functions**: camelCase (e.g., `ragas.ts`, `musicGeneration.ts`)
- **Schema**: camelCase (e.g., `schema.ts`)

### **ML Files**
- **Python files**: snake_case (e.g., `raga_classifier.py`, `cloud_run_app.py`)
- **Configuration**: descriptive names (e.g., `requirements_ml.txt`)

### **Documentation Files**
- **Markdown files**: UPPERCASE_WITH_UNDERSCORES (e.g., `ML_RAGA_DETECTION_SCIENTIFIC.md`)
- **Descriptive names**: Clear purpose indication

## **Configuration Files**

### **Frontend Configuration**
- `package.json` - Dependencies and scripts
- `lynx.config.ts` - Lynx build configuration
- `tailwind.config.js` - Tailwind CSS configuration
- `tsconfig.json` - TypeScript configuration

### **Backend Configuration**
- `convex/schema.ts` - Database schema definition
- Environment variables for Convex deployment

### **ML Configuration**
- `requirements_ml.txt` - Python ML dependencies
- `requirements_cloud_run.txt` - Cloud Run specific dependencies
- `Dockerfile` - Container configuration

## **Benefits of This Organization**

### **Developer Experience**
- **Clear structure** - Easy to find files and understand purpose
- **Logical grouping** - Related files grouped together
- **Consistent naming** - Predictable file and folder names
- **Separation of concerns** - Frontend, backend, and ML clearly separated

### **Maintainability**
- **Modular structure** - Easy to modify individual components
- **Clear dependencies** - Dependencies clearly defined
- **Documentation** - Comprehensive documentation for each component
- **Version control** - Clean git history with logical commits

### **Scalability**
- **Extensible structure** - Easy to add new features
- **Component reusability** - Reusable components and utilities
- **Clear interfaces** - Well-defined APIs and interfaces
- **Testing support** - Structure supports comprehensive testing

### **Collaboration**
- **Team-friendly** - Multiple developers can work simultaneously
- **Clear ownership** - Clear ownership of different components
- **Code review** - Easy to review changes in logical units
- **Documentation** - Comprehensive documentation for onboarding

## **Best Practices**

### **1. File Organization**
- Keep related files together
- Use descriptive folder names
- Maintain consistent structure across similar components

### **2. Naming Conventions**
- Follow established conventions for each technology
- Use descriptive names that indicate purpose
- Be consistent across the entire codebase

### **3. Documentation**
- Document all major components
- Keep documentation up to date
- Use clear, concise language

### **4. Version Control**
- Make logical, atomic commits
- Use descriptive commit messages
- Keep branches focused on specific features

## **Getting Started for New Developers**

### **1. Clone and Setup**
```bash
git clone <repository-url>
cd raga_detector
```

### **2. Install Dependencies**
```bash
# Frontend dependencies
cd frontend && bun install

# Python dependencies
cd .. && pip install -r requirements.txt
```

### **3. Start Development**
```bash
# Start all services
./run_dev.sh
```

### **4. Read Documentation**
- Start with `README.md` for project overview
- Read `docs/QUICK_DEPLOYMENT_GUIDE.md` for setup
- Check `docs/` folder for detailed documentation

## **Conclusion**

This organized codebase structure provides:

- **Clear navigation** - Easy to find and understand code
- **Developer-friendly** - Intuitive structure for new developers
- **Maintainable** - Easy to modify and extend
- **Scalable** - Supports growth and new features
- **Well-documented** - Comprehensive documentation for all components

**The codebase is now organized, clean, and ready for productive development!**
