# RagaSense Frontend Deployment Guide

> **Complete guide for deploying the Lynx frontend to various hosting platforms**

## 🎯 **Overview**

This guide covers deploying the RagaSense Lynx frontend to multiple hosting platforms. The frontend is built using the Lynx framework with Rspeedy and outputs to `dist/web` for web deployment.

## 📋 **Prerequisites**

1. **Built Frontend**: Ensure the frontend is built successfully
2. **Environment Variables**: Configure your Convex and API URLs
3. **Domain/Subdomain**: Optional but recommended for production

## 🚀 **Quick Start**

### **1. Build the Frontend**

```bash
cd frontend
bun install
bun run build:web
```

This creates the production build in `frontend/dist/web/`.

### **2. Test Locally**

```bash
cd frontend
bun run preview
```

Visit `http://localhost:3000` to verify the build works.

## 🌐 **Deployment Options**

### **Option 1: Netlify (Recommended for Static Sites)**

**Pros**: Free tier, automatic deployments, CDN, easy setup  
**Cons**: Limited server-side features  

#### **Setup Steps:**

1. **Install Netlify CLI**:
   ```bash
   npm install -g netlify-cli
   ```

2. **Deploy**:
   ```bash
   cd frontend
   netlify deploy --prod --dir=dist/web
   ```

3. **Or connect to Git**:
   - Push to GitHub
   - Connect Netlify to your repository
   - Set build command: `bun run build:web`
   - Set publish directory: `dist/web`

#### **Configuration** (`frontend/deploy/netlify.toml`):
```toml
[build]
  command = "bun run build:web"
  publish = "dist/web"
  base = "frontend"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### **Option 2: Vercel (Recommended for React/Next.js)**

**Pros**: Excellent performance, automatic deployments, edge functions  
**Cons**: More complex than Netlify  

#### **Setup Steps:**

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy**:
   ```bash
   cd frontend
   vercel --prod
   ```

3. **Or connect to Git**:
   - Push to GitHub
   - Connect Vercel to your repository
   - Vercel will auto-detect the build settings

#### **Configuration** (`frontend/deploy/vercel.json`):
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist/web",
        "buildCommand": "bun run build:web"
      }
    }
  ]
}
```

### **Option 3: Firebase Hosting (Google Cloud)**

**Pros**: Google's infrastructure, good performance, easy scaling  
**Cons**: Requires Google account, more complex setup  

#### **Setup Steps:**

1. **Install Firebase CLI**:
   ```bash
   npm install -g firebase-tools
   ```

2. **Initialize Firebase**:
   ```bash
   cd frontend
   firebase login
   firebase init hosting
   ```

3. **Deploy**:
   ```bash
   firebase deploy --only hosting
   ```

#### **Configuration** (`frontend/deploy/firebase.json`):
```json
{
  "hosting": {
    "public": "dist/web",
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ]
  }
}
```

### **Option 4: GitHub Pages (Free)**

**Pros**: Free, integrated with GitHub, automatic deployments  
**Cons**: Limited features, slower than CDN  

#### **Setup Steps:**

1. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Source: GitHub Actions

2. **Push the workflow**:
   - The GitHub Actions workflow is already configured
   - Push to main branch to trigger deployment

#### **Configuration** (`.github/workflows/deploy.yml`):
```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./frontend/dist/web
```

### **Option 5: Railway (Full-Stack)**

**Pros**: Full-stack platform, easy database integration  
**Cons**: Paid after free tier, more complex  

#### **Setup Steps:**

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy**:
   ```bash
   cd frontend
   railway login
   railway init
   railway up
   ```

#### **Configuration** (`frontend/deploy/railway.toml`):
```toml
[build]
builder = "nixpacks"
buildCommand = "cd frontend && bun install && bun run build:web"

[deploy]
startCommand = "cd frontend && bun run preview"
```

### **Option 6: Docker (Self-Hosted)**

**Pros**: Full control, can deploy anywhere  
**Cons**: Requires server management, more complex  

#### **Setup Steps:**

1. **Build Docker Image**:
   ```bash
   cd frontend
   docker build -f deploy/Dockerfile -t ragasense-frontend .
   ```

2. **Run Container**:
   ```bash
   docker run -p 80:80 -e VITE_CONVEX_URL=your-url ragasense-frontend
   ```

#### **Configuration** (`frontend/deploy/Dockerfile`):
```dockerfile
FROM oven/bun:1 as base
WORKDIR /app
COPY . .
RUN bun install && bun run build:web

FROM nginx:alpine
COPY --from=base /app/dist/web /usr/share/nginx/html
```

## 🔧 **Environment Configuration**

### **Required Environment Variables**

```bash
# Convex Database URL
VITE_CONVEX_URL=https://your-convex-deployment.convex.cloud

# Backend API URL (if using separate backend)
VITE_API_URL=https://your-backend-api.com

# Environment
NODE_ENV=production
```

### **Setting Environment Variables**

#### **Netlify**:
```bash
netlify env:set VITE_CONVEX_URL "https://your-convex-deployment.convex.cloud"
netlify env:set VITE_API_URL "https://your-backend-api.com"
```

#### **Vercel**:
```bash
vercel env add VITE_CONVEX_URL
vercel env add VITE_API_URL
```

#### **Firebase**:
```bash
firebase functions:config:set app.convex_url="https://your-convex-deployment.convex.cloud"
firebase functions:config:set app.api_url="https://your-backend-api.com"
```

#### **Railway**:
```bash
railway variables set VITE_CONVEX_URL="https://your-convex-deployment.convex.cloud"
railway variables set VITE_API_URL="https://your-backend-api.com"
```

## 🌍 **Custom Domain Setup**

### **Netlify**:
1. Go to Site Settings → Domain Management
2. Add custom domain
3. Update DNS records as instructed

### **Vercel**:
1. Go to Project Settings → Domains
2. Add custom domain
3. Update DNS records as instructed

### **Firebase**:
1. Go to Hosting → Custom Domains
2. Add custom domain
3. Update DNS records as instructed

## 📊 **Performance Optimization**

### **Build Optimization**

1. **Enable Compression**:
   ```bash
   # In rspeedy.config.ts
   build: {
     minify: true,
     sourcemap: false, // Disable in production
     target: 'es2020'
   }
   ```

2. **Asset Optimization**:
   - Images: Use WebP format
   - Fonts: Use `font-display: swap`
   - CSS: Enable PurgeCSS for Tailwind

### **CDN Configuration**

All platforms provide CDN by default. For custom CDN:

1. **Cloudflare**:
   - Add your domain to Cloudflare
   - Configure caching rules
   - Enable Brotli compression

2. **AWS CloudFront**:
   - Create distribution
   - Configure origin
   - Set up caching behaviors

## 🔒 **Security Headers**

All deployment configurations include security headers:

```http
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

## 📈 **Monitoring & Analytics**

### **Performance Monitoring**

1. **Web Vitals**:
   - Core Web Vitals tracking
   - Performance budgets
   - Real User Monitoring (RUM)

2. **Error Tracking**:
   - Sentry integration
   - Error boundaries
   - Log aggregation

### **Analytics**

1. **Google Analytics**:
   - Page views
   - User behavior
   - Conversion tracking

2. **Custom Analytics**:
   - Feature usage
   - Performance metrics
   - User engagement

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Build Fails**:
   ```bash
   # Check dependencies
   bun install
   
   # Clear cache
   rm -rf node_modules bun.lockb
   bun install
   
   # Check TypeScript errors
   bun run type-check
   ```

2. **Environment Variables Not Working**:
   ```bash
   # Check if variables are set
   echo $VITE_CONVEX_URL
   
   # Rebuild after setting variables
   bun run build:web
   ```

3. **Routing Issues**:
   - Ensure SPA routing is configured
   - Check redirect rules
   - Verify `index.html` fallback

4. **CORS Issues**:
   - Configure CORS in backend
   - Check API URLs
   - Verify domain whitelist

### **Debug Commands**

```bash
# Local development
cd frontend
bun run dev

# Production build test
bun run build:web
bun run preview

# Type checking
bun run type-check

# Linting
bun run lint
```

## 📚 **Additional Resources**

- [Netlify Documentation](https://docs.netlify.com/)
- [Vercel Documentation](https://vercel.com/docs)
- [Firebase Documentation](https://firebase.google.com/docs)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Railway Documentation](https://docs.railway.app/)
- [Docker Documentation](https://docs.docker.com/)

## 🎯 **Recommended Deployment Strategy**

### **For Development/Testing**:
- **GitHub Pages**: Free, integrated with repository
- **Netlify**: Easy setup, good for demos

### **For Production**:
- **Vercel**: Best performance, excellent developer experience
- **Netlify**: Good alternative, easier setup

### **For Enterprise**:
- **Firebase**: Google's infrastructure, good scaling
- **Docker**: Full control, deploy anywhere

---

**Last Updated**: January 2024  
**Maintained By**: RagaSense Development Team
