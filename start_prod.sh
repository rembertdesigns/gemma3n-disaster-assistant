#!/bin/bash
# Production startup script for Emergency Response Assistant

echo ""
echo "🚨 Starting Emergency Response Assistant (Production)"
echo "================================================================"
echo "Activating virtual environment..."
source '/Users/richpointofview/gemma3n-disaster-assistant/venv/bin/activate'

echo "Starting production server with 4 workers..."
echo ""
echo "🌐 Server will be available at: http://localhost:8000"
echo ""

python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
