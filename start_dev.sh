#!/bin/bash
# Development startup script for Emergency Response Assistant

echo ""
echo "🚨 Starting Emergency Response Assistant (Development)"
echo "================================================================"
echo "Activating virtual environment..."
source '/Users/richpointofview/gemma3n-disaster-assistant/venv/bin/activate'

echo "Starting development server..."
echo ""
echo "🌐 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/api/docs"  
echo "🏥 Health Check: http://localhost:8000/health"
echo ""

python api.py
