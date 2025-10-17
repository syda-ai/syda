#!/bin/bash

# Syda Full Stack Development Start Script

echo "🚀 Starting Syda Full Stack..."
echo ""
echo "This will start:"
echo "  - PostgreSQL on port 5432"
echo "  - API on http://localhost:8000"
echo "  - UI on http://localhost:5173"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Start services
echo "📦 Starting services with Docker Compose..."
docker-compose up --build

# Note: Ctrl+C will stop all services

