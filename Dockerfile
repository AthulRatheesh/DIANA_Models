FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y libgomp1

# Copy and install requirements
COPY recipe_api/recipe_requirements.txt .
COPY image_recog/image_requirements.txt .
RUN pip install --no-cache-dir -r recipe_requirements.txt -r image_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"

# Copy application files
COPY recipe_api/recipe_QA.py .
COPY image_recog/Image_recom_backend.py .
COPY main.py .

# Create models directory and copy models
RUN mkdir -p models
COPY recipe_api/models/recipe_qa_model.joblib models/
COPY image_recog/models/model.h5 models/
COPY image_recog/models/class_indices.json models/

# Set environment variables
ENV RECIPE_MODEL_PATH=/app/models/recipe_qa_model.joblib
ENV IMAGE_MODEL_PATH=/app/models/model.h5
ENV CLASSES_PATH=/app/models/class_indices.json
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
