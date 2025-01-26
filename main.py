from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from recipe_api.recipe_QA import RecipeQABackend
from image_recog.image_backend import ImagePredictor
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Union, Optional
import os

app = FastAPI(title="DIANA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backends
recipe_backend = None
image_predictor = None

class RecipeQuery(BaseModel):
    query: str
    num_results: Optional[int] = 3

class RecipeResponse(BaseModel):
    name: str
    ingredients: str
    directions: str
    prep_time: str
    cook_time: str
    total_time: str
    servings: Union[str, int]
    rating: Union[str, float]

@app.on_event("startup")
async def startup_event():
    global recipe_backend, image_predictor
    try:
        recipe_path = os.path.abspath(os.getenv("RECIPE_MODEL_PATH", "models/recipe_qa_model.joblib"))
        image_path = os.path.abspath(os.getenv("IMAGE_MODEL_PATH", "models/model.h5"))
        classes_path = os.path.abspath(os.getenv("CLASSES_PATH", "models/class_indices.json"))
        
        print(f"Loading models from: {recipe_path}, {image_path}")
        recipe_backend = RecipeQABackend(recipe_path)
        image_predictor = ImagePredictor(image_path, classes_path)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"status": "ok", "message": "DIANA API is running"}

@app.post("/api/v1/recipes/search", response_model=List[RecipeResponse])
async def search_recipes(query: RecipeQuery):
    if not recipe_backend:
        raise HTTPException(status_code=500, detail="Recipe model not initialized")
    try:
        return recipe_backend.get_recipe_response(query.query, query.num_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict")
async def predict_image(file: UploadFile = File(...)):
    if not image_predictor:
        raise HTTPException(status_code=500, detail="Image model not initialized")
    try:
        return await image_predictor.predict(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    if not recipe_backend or not image_predictor:
        raise HTTPException(status_code=503, detail="Services not ready")
    return {"status": "healthy"}
