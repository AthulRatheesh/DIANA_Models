import joblib
from typing import List, Dict, Union
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class RecipeQABackend:
    def __init__(self, model_path: str):
        """
        Initialize the Recipe QA Backend
        
        Args:
            model_path (str): Path to the saved .joblib model file
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
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
