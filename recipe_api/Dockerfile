FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY recipe_api/recipe_requirements.txt .
RUN pip install --no-cache-dir -r recipe_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy the rest of the application
COPY recipe_api/recipe_api.py .
COPY recipe_api/recipe_QA.py .

# Create models directory and copy model
RUN mkdir -p models
COPY recipe_api/models/recipe_qa_model.joblib models/

# Verify files and permissions
RUN ls -la /app && \
    ls -la /app/models && \
    python -c "import joblib; print('Model test:', joblib.load('/app/models/recipe_qa_model.joblib'))"

ENV MODEL_PATH=/app/models/recipe_qa_model.joblib
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "recipe_api:app", "--host", "0.0.0.0", "--port", "8000"]
