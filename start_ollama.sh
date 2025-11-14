#!/bin/bash
set -e
ollama serve &
sleep 8
ollama pull ${OLLAMA_MODEL:-pixtral} || true
wait
