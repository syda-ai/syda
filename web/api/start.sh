#!/bin/bash

# Syda API Start Script

echo "🚀 Starting Syda API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env with your configuration!"
fi

# Run database migrations
echo "🗄️  Running database migrations..."
alembic upgrade head

# Start the server
echo "✅ Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000

