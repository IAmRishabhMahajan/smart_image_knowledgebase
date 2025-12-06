#!/bin/bash
set -e

echo "Starting FastAPI (port 8000)..."
uvicorn src.app:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo "Starting Streamlit UI (port 8501)..."
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501 &
STREAMLIT_PID=$!

wait -n

kill $FASTAPI_PID $STREAMLIT_PID
