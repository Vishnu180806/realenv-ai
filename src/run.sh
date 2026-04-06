#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Start the FastAPI server
echo ""
echo "✨ NEBULA UI Console Starting ✨"
echo "-----------------------------------"
echo "🌐 Animated Dashboard: http://localhost:7860/"
echo "-----------------------------------"
echo "Press CTRL+C to stop the server."
echo ""

python3 app.py
