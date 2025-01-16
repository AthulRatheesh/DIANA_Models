from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recipe_backend import RecipeQABackend
import os
from typing import List, Dict, Union, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DIANA Recipe API",
    description="API for recipe recommendations and queries",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the backend
qa_backend = None

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
    global qa_backend
    try:
        model_path = os.getenv("MODEL_PATH", "models/recipe_qa_model.joblib")
        qa_backend = RecipeQABackend(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"status": "ok", "message": "DIANA Recipe API is running"}

@app.post("/api/v1/recipes/search", response_model=List[RecipeResponse])
async def search_recipes(query: RecipeQuery):
    if not qa_backend:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        results = qa_backend.get_recipe_response(
            query.query,
            num_results=query.num_results
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    if not qa_backend:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy"}
