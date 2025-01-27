FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1

# Copy and install requirements
COPY recipe_api/recipe_requirements.txt recipe_api/
COPY image_recog/image_requirements.txt image_recog/
RUN pip install --no-cache-dir -r recipe_api/recipe_requirements.txt -r image_recog/image_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger');nltk.download('punkt_tab')"

# Copy directories maintaining structure
COPY recipe_api/ recipe_api/
COPY image_recog/ image_recog/
COPY main.py .

ENV RECIPE_MODEL_PATH=/app/recipe_api/models/recipe_qa_model.joblib
ENV IMAGE_MODEL_PATH=/app/image_recog/models/model.h5
ENV CLASSES_PATH=/app/image_recog/models/class_indices.json
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
