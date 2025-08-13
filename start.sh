#!/bin/bash

# Data Analyst Agent API Startup Script

echo "🚀 Starting Data Analyst Agent API..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before running the server."
fi

# Run the API
echo "🌐 Starting the API server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "🏥 Health check at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
