FROM python:3.11-slim

WORKDIR /app

# System deps (opencv, etc. if you need them)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .

# Add Streamlit + requests for the UI
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose both ports (FastAPI + Streamlit)
EXPOSE 8000
EXPOSE 8501

# Default command (used only if docker-compose doesn't override)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
