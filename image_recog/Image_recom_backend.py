# main.py (FastAPI)
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
import io

app = FastAPI()

# Load model and classes
model = load_model('/models/model.h5')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_idx]
    confidence = float(prediction[0][predicted_idx])
    
    return {"class": predicted_class, "confidence": confidence}
