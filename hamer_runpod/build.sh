#!/bin/bash
# Build HaMeR Docker images with smart layering
# Usage:
#   ./build.sh base     - Build base image (heavy, ~30 min, do once)
#   ./build.sh handler  - Build handler layer (fast, ~1 min)
#   ./build.sh push     - Push latest to Docker Hub

set -e

IMAGE_NAME="naman188/hamer-runpod"

case "$1" in
    base)
        echo "=========================================="
        echo "Building BASE image (this takes ~30 min)"
        echo "=========================================="
        docker buildx build \
            --platform linux/amd64 \
            -f Dockerfile.base \
            -t $IMAGE_NAME:base \
            --load \
            .
        echo ""
        echo "✅ Base image built: $IMAGE_NAME:base"
        echo "Now run: ./build.sh handler"
        ;;
    
    handler)
        echo "=========================================="
        echo "Building HANDLER layer (fast)"
        echo "=========================================="
        docker buildx build \
            --platform linux/amd64 \
            -f Dockerfile.handler \
            -t $IMAGE_NAME:latest \
            --load \
            .
        echo ""
        echo "✅ Handler image built: $IMAGE_NAME:latest"
        echo "Now run: ./build.sh push"
        ;;
    
    push)
        echo "Pushing to Docker Hub..."
        docker push $IMAGE_NAME:base
        docker push $IMAGE_NAME:latest
        echo ""
        echo "✅ Pushed both images!"
        echo "Use in RunPod: $IMAGE_NAME:latest"
        ;;
    
    all)
        $0 base
        $0 handler
        $0 push
        ;;
    
    *)
        echo "Usage: ./build.sh [base|handler|push|all]"
        echo ""
        echo "  base    - Build base image with models (~30 min, do once)"
        echo "  handler - Build handler layer (~1 min, do often)"
        echo "  push    - Push images to Docker Hub"
        echo "  all     - Do all three"
        exit 1
        ;;
esac
