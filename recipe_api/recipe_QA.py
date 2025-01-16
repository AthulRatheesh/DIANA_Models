import joblib
from typing import List, Dict, Union
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

class RecipeQABackend:
    def __init__(self, model_path: str):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("recipe_QA")
    
        try:
        # Verify input
            if not isinstance(model_path, str):
                raise ValueError(f"Expected string path, got {type(model_path)}: {model_path}")
            
        # Verify file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            # Load the saved model components
            self.logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)
            
            # Extract components
            self.vectorizer = model_data['vectorizer']
            self.recipe_vectors = model_data['recipe_vectors']
            self.recipes_df = model_data['recipes_df']
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            self.logger.info("Model components loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
        tokens = word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]
        return ' '.join(tokens)

    def get_recipe_response(self, query: str, num_results: int = 3) -> List[Dict[str, Union[str, float]]]:
        """
        Get recipe recommendations for a query
        
        Args:
            query (str): User's recipe query
            num_results (int): Number of recipes to return
            
        Returns:
            List[Dict]: List of recipe recommendations
        """
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.recipe_vectors).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[::-1][:num_results]
            
            # Format results
            formatted_results = []
            for idx in top_indices:
                recipe = self.recipes_df.iloc[idx]
                formatted_recipe = {
                    'name': recipe['recipe_name'],
                    'ingredients': recipe['ingredients'],
                    'directions': recipe['directions'],
                    'prep_time': recipe.get('prep_time', 'Not specified'),
                    'cook_time': recipe.get('cook_time', 'Not specified'),
                    'total_time': recipe.get('total_time', 'Not specified'),
                    'servings': recipe.get('servings', 'Not specified'),
                    'rating': recipe.get('rating', 'Not rated'),
                }
                formatted_results.append(formatted_recipe)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error getting recipe recommendations: {str(e)}")
            return []
