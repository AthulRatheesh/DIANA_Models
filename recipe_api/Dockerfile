# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY recipe_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy the rest of the application
COPY recipe_api.py .
COPY recipe_QA.py .
COPY models/ models/

# Set environment variables
ENV MODEL_PATH=models/QA_model.joblib
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
version: '3.8'

services:
  recipe-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=models/recipe_qa_model.joblib
      - PORT=8000
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
