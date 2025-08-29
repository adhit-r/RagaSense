#!/bin/bash

# RagaSense Frontend Deployment Script
# Usage: ./deploy.sh [platform]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to build the frontend
build_frontend() {
    print_status "Building frontend..."
    
    if ! command_exists bun; then
        print_error "Bun is not installed. Please install Bun first."
        exit 1
    fi
    
    cd frontend
    bun install
    bun run build:web
    
    if [ $? -eq 0 ]; then
        print_success "Frontend built successfully!"
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Function to deploy to Netlify
deploy_netlify() {
    print_status "Deploying to Netlify..."
    
    if ! command_exists netlify; then
        print_warning "Netlify CLI not found. Installing..."
        npm install -g netlify-cli
    fi
    
    cd frontend
    netlify deploy --prod --dir=dist/web
    
    print_success "Deployed to Netlify!"
}

# Function to deploy to Vercel
deploy_vercel() {
    print_status "Deploying to Vercel..."
    
    if ! command_exists vercel; then
        print_warning "Vercel CLI not found. Installing..."
        npm install -g vercel
    fi
    
    cd frontend
    vercel --prod
    
    print_success "Deployed to Vercel!"
}

# Function to deploy to Firebase
deploy_firebase() {
    print_status "Deploying to Firebase..."
    
    if ! command_exists firebase; then
        print_warning "Firebase CLI not found. Installing..."
        npm install -g firebase-tools
    fi
    
    cd frontend
    firebase deploy --only hosting
    
    print_success "Deployed to Firebase!"
}

# Function to deploy to GitHub Pages
deploy_github_pages() {
    print_status "Deploying to GitHub Pages..."
    
    print_warning "GitHub Pages deployment requires pushing to main branch."
    print_warning "Please ensure you have the GitHub Actions workflow configured."
    
    # Check if we're on main branch
    if [ "$(git branch --show-current)" != "main" ]; then
        print_error "You must be on the main branch to deploy to GitHub Pages."
        print_status "Please merge your changes to main first."
        exit 1
    fi
    
    # Push to trigger GitHub Actions
    git push origin main
    
    print_success "Triggered GitHub Pages deployment!"
}

# Function to deploy with Docker
deploy_docker() {
    print_status "Building and deploying with Docker..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    cd frontend
    
    # Build Docker image
    docker build -f deploy/Dockerfile -t ragasense-frontend .
    
    # Stop existing container if running
    docker stop ragasense-frontend 2>/dev/null || true
    docker rm ragasense-frontend 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name ragasense-frontend \
        -p 80:80 \
        -e VITE_CONVEX_URL="${VITE_CONVEX_URL:-https://your-convex-deployment.convex.cloud}" \
        -e VITE_API_URL="${VITE_API_URL:-https://your-backend-api.com}" \
        ragasense-frontend
    
    print_success "Deployed with Docker! Access at http://localhost"
}

# Function to deploy to Railway
deploy_railway() {
    print_status "Deploying to Railway..."
    
    if ! command_exists railway; then
        print_warning "Railway CLI not found. Installing..."
        npm install -g @railway/cli
    fi
    
    cd frontend
    railway up
    
    print_success "Deployed to Railway!"
}

# Function to show help
show_help() {
    echo "RagaSense Frontend Deployment Script"
    echo ""
    echo "Usage: $0 [platform]"
    echo ""
    echo "Platforms:"
    echo "  netlify     - Deploy to Netlify (recommended for static sites)"
    echo "  vercel      - Deploy to Vercel (best performance)"
    echo "  firebase    - Deploy to Firebase Hosting"
    echo "  github      - Deploy to GitHub Pages"
    echo "  docker      - Deploy with Docker (self-hosted)"
    echo "  railway     - Deploy to Railway"
    echo "  build       - Only build, don't deploy"
    echo "  help        - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VITE_CONVEX_URL  - Your Convex deployment URL"
    echo "  VITE_API_URL     - Your backend API URL"
    echo ""
    echo "Examples:"
    echo "  $0 netlify"
    echo "  $0 vercel"
    echo "  VITE_CONVEX_URL=https://my-app.convex.cloud $0 docker"
}

# Main script
main() {
    local platform=${1:-help}
    
    case $platform in
        netlify)
            build_frontend
            deploy_netlify
            ;;
        vercel)
            build_frontend
            deploy_vercel
            ;;
        firebase)
            build_frontend
            deploy_firebase
            ;;
        github)
            build_frontend
            deploy_github_pages
            ;;
        docker)
            build_frontend
            deploy_docker
            ;;
        railway)
            build_frontend
            deploy_railway
            ;;
        build)
            build_frontend
            print_success "Build completed! Files are in frontend/dist/web/"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown platform: $platform"
            show_help
            exit 1
            ;;
    esac
}

# Check if script is run from the correct directory
if [ ! -f "frontend/package.json" ]; then
    print_error "Please run this script from the project root directory."
    exit 1
fi

# Run main function
main "$@"
