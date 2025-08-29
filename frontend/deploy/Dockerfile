# Multi-stage build for production
FROM oven/bun:1 as base
WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/bun.lockb ./
RUN bun install --frozen-lockfile

# Copy source code
COPY frontend/ ./

# Build for production
RUN bun run build:web

# Production stage
FROM nginx:alpine

# Copy built assets
COPY --from=base /app/dist/web /usr/share/nginx/html

# Copy nginx configuration
COPY frontend/deploy/nginx.conf /etc/nginx/nginx.conf

# Copy environment variables
COPY frontend/deploy/env.sh /docker-entrypoint.d/40-env.sh
RUN chmod +x /docker-entrypoint.d/40-env.sh

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
