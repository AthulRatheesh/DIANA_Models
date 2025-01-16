FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY recipe_api/recipe_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r recipe_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy the rest of the application
COPY recipe_api/recipe_api.py .
COPY recipe_api/recipe_QA.py .
COPY recipe_api/models/ models/

# Set environment variables
ENV MODEL_PATH=models/QAModel.joblib
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
