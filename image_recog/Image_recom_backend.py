# image_recom_backend.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
import io
import os

class ImagePredictor:
   def __init__(self, model_path: str, classes_path: str):
       """Initialize the image predictor with model and classes"""
       self.model = load_model(model_path)
       with open(classes_path, 'r') as f:
           self.class_indices = json.load(f)
       self.class_names = [k for k, v in sorted(self.class_indices.items(), key=lambda x: x[1])]

   async def predict(self, file):
       """Predict class and confidence for uploaded image"""
       contents = await file.read()
       img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
       img_array = image.img_to_array(img)
       img_array = np.expand_dims(img_array, axis=0)
       img_array = preprocess_input(img_array)
       
       prediction = self.model.predict(img_array)
       predicted_idx = np.argmax(prediction[0])
       predicted_class = self.class_names[predicted_idx]
       confidence = float(prediction[0][predicted_idx])
       
       return {"class": predicted_class, "confidence": confidence}
