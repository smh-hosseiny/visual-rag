#!/bin/bash

# conda activate llm-ocr
echo "🚀 Starting MarketLens API Server..."
echo "✅ Environment: llm-ocr"
echo "📡 Listening on: http://0.0.0.0:8001"

# Run Uvicorn with reload enabled for development
uvicorn app.api:app --host 0.0.0.0 --port 8001 --reload


