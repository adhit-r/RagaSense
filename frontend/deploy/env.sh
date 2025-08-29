#!/bin/sh

# Replace environment variables in the built files
# This script runs before nginx starts

echo "Configuring environment variables..."

# Replace placeholders in index.html
if [ -n "$VITE_CONVEX_URL" ]; then
    sed -i "s|https://your-convex-deployment.convex.cloud|$VITE_CONVEX_URL|g" /usr/share/nginx/html/index.html
fi

if [ -n "$VITE_API_URL" ]; then
    sed -i "s|https://your-backend-api.com|$VITE_API_URL|g" /usr/share/nginx/html/index.html
fi

# Replace placeholders in nginx config
if [ -n "$VITE_CONVEX_URL" ]; then
    sed -i "s|https://your-convex-deployment.convex.cloud|$VITE_CONVEX_URL|g" /etc/nginx/nginx.conf
fi

if [ -n "$VITE_API_URL" ]; then
    sed -i "s|https://your-backend-api.com|$VITE_API_URL|g" /etc/nginx/nginx.conf
fi

echo "Environment configuration complete."
