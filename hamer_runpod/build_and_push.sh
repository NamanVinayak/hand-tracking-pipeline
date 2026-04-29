#!/bin/bash
# Build and push incremental HaMeR update
# Uses existing base image, only updates handler.py

set -e

IMAGE_NAME="naman188/hamer-runpod"
TAG="v2-r2"

echo "=========================================="
echo "Building incremental update for Cloudflare R2"
echo "Base: $IMAGE_NAME:latest"
echo "New:  $IMAGE_NAME:$TAG"
echo "=========================================="

# Build using the incremental Dockerfile
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.update \
    -t $IMAGE_NAME:$TAG \
    --load \
    .

echo ""
echo "Build complete! Image size:"
docker images | grep $IMAGE_NAME

echo ""
echo "Pushing to Docker Hub..."
docker push $IMAGE_NAME:$TAG

echo ""
echo "=========================================="
echo "SUCCESS!"
echo "Image: $IMAGE_NAME:$TAG"
echo "=========================================="
echo ""
echo "RunPod Environment Secrets needed:"
echo "  S3_ENDPOINT_URL     = https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
echo "  S3_ACCESS_KEY_ID    = <from Cloudflare R2>"
echo "  S3_SECRET_ACCESS_KEY = <from Cloudflare R2>"
echo "  S3_REGION           = auto"
