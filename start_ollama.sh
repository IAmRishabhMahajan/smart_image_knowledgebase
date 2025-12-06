#!/bin/bash
set -e

# Start Ollama server
ollama serve &
sleep 8

# Pull a REAL vision model
ollama pull llava || true

wait
