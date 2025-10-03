#!/bin/bash
# Build script for ML4T Trading System Docker containers

set -e

echo "🐳 Building ML4T Trading System Docker Images"
echo "=============================================="

# Build arguments
APP_VERSION=${APP_VERSION:-latest}
BUILD_ENV=${BUILD_ENV:-production}

echo "📦 Building base image (builder stage)..."
docker build \
    --target builder \
    --build-arg BUILD_ENV=$BUILD_ENV \
    -t ml4t-trading:builder-$APP_VERSION \
    .

echo "🚀 Building production image (runtime stage)..."
docker build \
    --target runtime \
    --build-arg APP_ENV=$BUILD_ENV \
    --build-arg APP_VERSION=$APP_VERSION \
    -t ml4t-trading:$APP_VERSION \
    -t ml4t-trading:latest \
    .

echo "✅ Build completed successfully!"
echo ""
echo "Available images:"
docker images | grep ml4t-trading

echo ""
echo "🚀 To run the system:"
echo "docker-compose up -d"
echo ""
echo "📊 For development:"
echo "docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d"