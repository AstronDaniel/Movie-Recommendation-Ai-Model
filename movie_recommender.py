"""
Movie Recommendation System
Implements content-based and collaborative filtering approaches
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except (ImportError, Exception) as e:
    # Check for common Windows DLL error
    error_str = str(e)
    print(f"TensorFlow import error: {error_str}")
    if "DLL load failed" in error_str:
        print("\n[WARNING] TensorFlow could not be loaded (DLL load failed).")
        print("To fix this, install the 'Microsoft Visual C++ Redistributable' for Visual Studio 2015-2022.")
        print("Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("Continuing without AI features...\n")
    else:
        print(f"Warning: TensorFlow could not be imported. AI features will be disabled. Error: {e}")
    TF_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional
from movie_api import MovieAPI


class MovieRecommender:
    """Movie recommendation system supporting both cold start and experienced users"""
    
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.user_ratings = {}
        # API key is loaded from environment variable in MovieAPI
        self.movie_api = MovieAPI()
        self.poster_cache = {}  # Cache for poster URLs
        
        # AI Model attributes
        self.ai_model = None
        self.user_encoder = None
        self.movie_encoder = None
        self.genre_encoder = None
        self.model_movies_df = None
        self.model_loaded = False
        self.model_load_error = None
        
        self._initialize_movies()
        self._load_posters()
        self._load_trained_model()

    def _load_trained_model(self):
        """Load the trained Keras model and encoders"""
        if not TF_AVAILABLE:
            self.model_load_error = "TensorFlow is not available (DLL load failed). Please install MS Visual C++ Redistributable."
            print(f"Skipping model loading: {self.model_load_error}")
            self.model_loaded = False
            return

        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'model_training')
            
            # Check if files exist
            required_files = ['movie_model.h5', 'user_ids.npy', 'movie_ids.npy', 'genre_ids.npy', 'movies_processed.csv']
            missing_files = []
            for f in required_files:
                if not os.path.exists(os.path.join(model_dir, f)):
                    missing_files.append(f)
            
            if missing_files:
                self.model_load_error = f"Missing model files: {', '.join(missing_files)}"
                print(self.model_load_error)
                self.model_loaded = False
                return

            # Load Model
            self.ai_model = tf.keras.models.load_model(os.path.join(model_dir, 'movie_model.h5'))
            
            # Load Encoders
            self.user_encoder = LabelEncoder()
            self.user_encoder.classes_ = np.load(os.path.join(model_dir, 'user_ids.npy'), allow_pickle=True)
            
            self.movie_encoder = LabelEncoder()
            self.movie_encoder.classes_ = np.load(os.path.join(model_dir, 'movie_ids.npy'), allow_pickle=True)
            
            self.genre_encoder = LabelEncoder()
            self.genre_encoder.classes_ = np.load(os.path.join(model_dir, 'genre_ids.npy'), allow_pickle=True)
            
            # Load Processed Movies
            self.model_movies_df = pd.read_csv(os.path.join(model_dir, 'movies_processed.csv'))
            
            # Pre-calculate normalized year for model input
            if 'year' in self.model_movies_df.columns:
                min_year = self.model_movies_df['year'].min()
                max_year = self.model_movies_df['year'].max()
                self.model_movies_df['year_norm'] = (self.model_movies_df['year'] - min_year) / (max_year - min_year)
            
            # Encode genres and movies for batch prediction
            # We need to map the 'primary_genre' to the encoded value
            # Note: LabelEncoder transforms based on the classes_ loaded
            
            # Create a mapping for faster lookups
            self.genre_map = {label: i for i, label in enumerate(self.genre_encoder.classes_)}
            self.movie_map = {label: i for i, label in enumerate(self.movie_encoder.classes_)}
            
            # Add encoded columns to model_movies_df
            self.model_movies_df['genre_encoded'] = self.model_movies_df['primary_genre'].map(self.genre_map).fillna(0).astype(int)
            
            # Filter out movies that might not be in the encoder (shouldn't happen if files are consistent)
            self.model_movies_df = self.model_movies_df[self.model_movies_df['movieId'].isin(self.movie_encoder.classes_)]
            self.model_movies_df['movie_encoded'] = self.model_movies_df['movieId'].map(self.movie_map).astype(int)
            
            self.model_loaded = True
            self.model_load_error = None
            print("AI Model loaded successfully!")
            
        except Exception as e:
            self.model_load_error = f"Error loading AI model: {str(e)}"
            print(self.model_load_error)
            self.model_loaded = False
        
    def _initialize_movies(self):
        """Initialize with popular movies from TMDB"""
        # Fetch popular movies
        popular_movies = self.movie_api.get_popular_movies(page=1)
        
        if not popular_movies:
            # Fallback to empty DataFrame with correct columns if API fails
            self.movies_df = pd.DataFrame(columns=[
                'movie_id', 'title', 'genre', 'year', 'rating', 
                'director', 'description', 'content', 'poster_path', 'backdrop_path'
            ])
            self.tfidf_matrix = None
            self.cosine_sim = None
            return

        self.movies_df = pd.DataFrame(popular_movies)
        
        # Ensure all required columns exist
        required_columns = ['movie_id', 'title', 'genre', 'year', 'rating', 'director', 'description']
        for col in required_columns:
            if col not in self.movies_df.columns:
                self.movies_df[col] = ''
                
        self._update_content_features()

    def _update_content_features(self):
        """Update TF-IDF and Cosine Similarity matrices"""
        if self.movies_df is None or self.movies_df.empty:
            return

        # Create content features by combining genre, director, and description
        # Handle missing values
        self.movies_df['genre'] = self.movies_df['genre'].fillna('')
        self.movies_df['director'] = self.movies_df['director'].fillna('Unknown')
        self.movies_df['description'] = self.movies_df['description'].fillna('')
        
        self.movies_df['content'] = (
            self.movies_df['genre'] + ' ' + 
            self.movies_df['director'] + ' ' + 
            self.movies_df['description']
        )
        
        # Create TF-IDF matrix for content-based filtering
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
            # Compute cosine similarity matrix
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        except ValueError:
            # Handle case with empty vocabulary or other issues
            self.tfidf_matrix = None
            self.cosine_sim = None
    
    def _load_posters(self):
        """Load movie poster URLs from API"""
        if self.movies_df is not None:
            movies_list = self.movies_df.to_dict('records')
            poster_urls = self.movie_api.get_poster_url_batch(movies_list)
            self.poster_cache = poster_urls
        
    def get_poster_url(self, title: str) -> str:
        """
        Get poster URL for a movie
        
        Args:
            title: Movie title
            
        Returns:
            Poster URL or None if not found
        """
        # Check cache first
        if title in self.poster_cache:
            return self.poster_cache[title]
        
        # Try to fetch from API
        movie_data = self.movies_df[self.movies_df['title'] == title]
        if not movie_data.empty:
            year = movie_data.iloc[0]['year']
            poster_url = self.movie_api.get_movie_poster_url(title, year)
            if poster_url:
                self.poster_cache[title] = poster_url
                return poster_url
        
        return None
        
    def get_all_movies(self) -> pd.DataFrame:
        """Return all movies in the dataset"""
        return self.movies_df[['movie_id', 'title', 'genre', 'year', 'rating', 'director', 'description']]
    
    def search_movies(self, query: str) -> pd.DataFrame:
        """Search for movies by title using TMDB API"""
        if not query:
            return self.get_all_movies()
        
        # Search via API
        results = self.movie_api.search_movies(query)
        
        if not results:
            return pd.DataFrame()
            
        # Add new movies to our dataset
        new_movies = []
        existing_titles = set(self.movies_df['title'].values) if self.movies_df is not None else set()
        
        for movie in results:
            if movie['title'] not in existing_titles:
                new_movies.append(movie)
        
        if new_movies:
            new_df = pd.DataFrame(new_movies)
            self.movies_df = pd.concat([self.movies_df, new_df], ignore_index=True)
            self._update_content_features()
            
        # Filter and return results
        mask = self.movies_df['title'].str.contains(query, case=False, na=False)
        return self.movies_df[mask][['movie_id', 'title', 'genre', 'year', 'rating', 'director', 'description']]
    
    def get_content_based_recommendations(
        self, 
        movie_title: str, 
        n_recommendations: int = 5
    ) -> List[Dict]:
        """
        Get recommendations based on movie content similarity
        Useful for cold start problem
        """
        # Find the movie index
        idx = self.movies_df[self.movies_df['title'] == movie_title].index
        
        if len(idx) == 0:
            return []
        
        idx = idx[0]
        
        # Get similarity scores for all movies with this movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top n most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations + 1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movies with similarity scores
        recommendations = []
        for idx, score in sim_scores:
            movie = self.movies_df.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'genre': movie['genre'],
                'year': movie['year'],
                'rating': movie['rating'],
                'director': movie['director'],
                'description': movie.get('description', 'No description available.'),
                'similarity_score': round(score * 100, 2)
            })
        
        return recommendations
    
    def add_user_rating(self, user_id: str, movie_title: str, rating: float):
        """Add a user rating for collaborative filtering"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        self.user_ratings[user_id][movie_title] = rating
    
    def get_collaborative_recommendations(
        self, 
        user_id: str, 
        n_recommendations: int = 5
    ) -> List[Dict]:
        """
        Get recommendations based on user ratings and similar users
        For experienced users with rating history
        """
        if user_id not in self.user_ratings or len(self.user_ratings[user_id]) == 0:
            # Fall back to top-rated movies for new users
            top_movies = self.movies_df.nlargest(n_recommendations, 'rating')
            return [
                {
                    'title': row['title'],
                    'genre': row['genre'],
                    'year': row['year'],
                    'rating': row['rating'],
                    'director': row['director'],
                    'description': row.get('description', 'No description available.'),
                    'recommendation_reason': 'Top Rated'
                }
                for _, row in top_movies.iterrows()
            ]
        
        # Get user's rated movies
        user_rated_movies = set(self.user_ratings[user_id].keys())
        
        # Find unrated movies
        all_movies = set(self.movies_df['title'].values)
        unrated_movies = all_movies - user_rated_movies
        
        # Score unrated movies based on similarity to highly rated movies
        movie_scores = {}
        for movie_title in unrated_movies:
            score = 0
            count = 0
            for rated_movie, user_rating in self.user_ratings[user_id].items():
                if user_rating >= 4.0:  # Consider movies rated 4.0 or higher
                    # Get similarity between movies
                    rated_idx = self.movies_df[self.movies_df['title'] == rated_movie].index
                    unrated_idx = self.movies_df[self.movies_df['title'] == movie_title].index
                    
                    if len(rated_idx) > 0 and len(unrated_idx) > 0:
                        similarity = self.cosine_sim[rated_idx[0]][unrated_idx[0]]
                        score += similarity * user_rating
                        count += 1
            
            if count > 0:
                movie_scores[movie_title] = score / count
        
        # Sort by score and get top recommendations
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_movies[:n_recommendations]
        
        recommendations = []
        for movie_title, score in top_movies:
            movie = self.movies_df[self.movies_df['title'] == movie_title].iloc[0]
            recommendations.append({
                'title': movie['title'],
                'genre': movie['genre'],
                'year': movie['year'],
                'rating': movie['rating'],
                'director': movie['director'],
                'description': movie.get('description', 'No description available.'),
                'recommendation_score': round(score * 100, 2)
            })
        
        return recommendations
    
    def load_more_popular_movies(self, page: int) -> pd.DataFrame:
        """Fetch and append more popular movies from TMDB"""
        new_movies_data = self.movie_api.get_popular_movies(page=page)
        
        if not new_movies_data:
            return pd.DataFrame()
            
        new_movies = []
        existing_titles = set(self.movies_df['title'].values) if self.movies_df is not None else set()
        
        for movie in new_movies_data:
            if movie['title'] not in existing_titles:
                new_movies.append(movie)
        
        if new_movies:
            new_df = pd.DataFrame(new_movies)
            # Ensure columns match
            for col in self.movies_df.columns:
                if col not in new_df.columns:
                    new_df[col] = ''
            
            # Only keep columns that exist in movies_df to avoid mismatch
            new_df = new_df[self.movies_df.columns]
            
            self.movies_df = pd.concat([self.movies_df, new_df], ignore_index=True)
            self._update_content_features()
            return new_df
            
        return pd.DataFrame()

    def get_hybrid_recommendations(
        self, 
        user_id: str, 
        movie_title: str = None, 
        n_recommendations: int = 5
    ) -> List[Dict]:
        """
        Hybrid approach combining content-based and collaborative filtering
        """
        if movie_title:
            # Use content-based filtering
            return self.get_content_based_recommendations(movie_title, n_recommendations)
        else:
            # Use collaborative filtering or top-rated for cold start
            return self.get_collaborative_recommendations(user_id, n_recommendations)
    
    def get_genres(self) -> List[str]:
        """Get list of all unique genres available in the dataset"""
        if self.movies_df is None or self.movies_df.empty:
            return []
        
        # Common TMDB genres
        common_genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
            "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", 
            "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
        ]
        
        # Filter to only include genres that actually exist in our dataframe
        existing_genres = []
        for genre in common_genres:
            if self.movies_df['genre'].str.contains(genre, case=False).any():
                existing_genres.append(genre)
                
        return sorted(existing_genres)

    def filter_movies(
        self, 
        genres: List[str] = None, 
        year_range: Tuple[int, int] = None, 
        rating_range: Tuple[float, float] = None,
        fetch_from_api: bool = True
    ) -> pd.DataFrame:
        """Filter movies by genre, year, and rating, optionally fetching more from API"""
        
        # If requested, fetch more movies from API matching criteria
        if fetch_from_api:
            try:
                # Convert genre names to IDs
                genre_ids = []
                if genres:
                    genre_name_map = self.movie_api.get_genre_id_map()
                    for g in genres:
                        # Handle potential case mismatches or partial matches
                        for name, gid in genre_name_map.items():
                            if g.lower() == name.lower():
                                genre_ids.append(gid)
                                break
                
                # Fetch from API
                min_rating = rating_range[0] if rating_range else None
                new_movies_data = self.movie_api.discover_movies(
                    genre_ids=genre_ids,
                    year_range=year_range,
                    rating_min=min_rating
                )
                
                # Add to dataframe
                if new_movies_data:
                    new_movies = []
                    existing_titles = set(self.movies_df['title'].values) if self.movies_df is not None else set()
                    
                    for movie in new_movies_data:
                        if movie['title'] not in existing_titles:
                            new_movies.append(movie)
                    
                    if new_movies:
                        new_df = pd.DataFrame(new_movies)
                        # Ensure columns match
                        for col in self.movies_df.columns:
                            if col not in new_df.columns:
                                new_df[col] = ''
                        new_df = new_df[self.movies_df.columns]
                        
                        self.movies_df = pd.concat([self.movies_df, new_df], ignore_index=True)
                        self._update_content_features()
            except Exception as e:
                print(f"Error fetching filtered movies from API: {e}")

        if self.movies_df is None or self.movies_df.empty:
            return pd.DataFrame()
            
        mask = pd.Series(True, index=self.movies_df.index)
        
        # Filter by genre (OR logic - if movie has ANY of the selected genres)
        if genres:
            genre_mask = pd.Series(False, index=self.movies_df.index)
            for genre in genres:
                genre_mask |= self.movies_df['genre'].str.contains(genre, case=False, na=False)
            mask &= genre_mask
            
        # Filter by year
        if year_range:
            min_year, max_year = year_range
            # Convert year to numeric, coercing errors to NaN
            years = pd.to_numeric(self.movies_df['year'], errors='coerce')
            mask &= (years >= min_year) & (years <= max_year)
            
        # Filter by rating
        if rating_range:
            min_rating, max_rating = rating_range
            ratings = pd.to_numeric(self.movies_df['rating'], errors='coerce')
            mask &= (ratings >= min_rating) & (ratings <= max_rating)
            
        return self.movies_df[mask][['movie_id', 'title', 'genre', 'year', 'rating', 'director', 'description']]

    def get_random_movies(self, n: int = 10) -> List[Dict]:
        """Get n random movies from the dataset"""
        if self.movies_df is None or self.movies_df.empty:
            return []
            
        n = min(n, len(self.movies_df))
        random_movies = self.movies_df.sample(n=n)
        
    def get_ai_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations using the trained Neural Collaborative Filtering model.
        
        Args:
            user_id: The ID of the user (from the training set) to predict for.
            n_recommendations: Number of movies to return.
        """
        if not self.model_loaded or not TF_AVAILABLE:
            return []
            
        try:
            # Check if user_id exists in encoder
            if user_id not in self.user_encoder.classes_:
                print(f"User {user_id} not found in training data.")
                return []
                
            # Encode User ID
            user_encoded = np.where(self.user_encoder.classes_ == user_id)[0][0]
            
            # Prepare Inputs for ALL movies
            n_movies = len(self.model_movies_df)
            
            # 1. User Input (Repeated)
            user_input = np.full(n_movies, user_encoded)
            
            # 2. Movie Input
            movie_input = self.model_movies_df['movie_encoded'].values
            
            # 3. Genre Input
            genre_input = self.model_movies_df['genre_encoded'].values
            
            # 4. Year Input
            year_input = self.model_movies_df['year_norm'].values
            
            # Predict
            predictions = self.ai_model.predict(
                [user_input, movie_input, genre_input, year_input], 
                batch_size=1024, 
                verbose=0
            )
            
            # Add predictions to dataframe
            results_df = self.model_movies_df.copy()
            results_df['predicted_rating'] = predictions.flatten()
            
            # Sort by prediction
            top_movies = results_df.nlargest(n_recommendations, 'predicted_rating')
            
            # Format results
            recommendations = []
            for _, row in top_movies.iterrows():
                title = row['title']
                # Try to get poster from our main app cache or API
                poster_url = self.get_poster_url(title)
                
                recommendations.append({
                    'title': title,
                    'genre': row['genres'].replace('|', ', '),
                    'year': int(row['year']),
                    'rating': round(float(row['predicted_rating']), 1), # Use predicted rating
                    'director': 'Unknown', # Not in training data
                    'description': 'AI Recommended Movie', # We could fetch this if we wanted
                    'poster_url': poster_url,
                    'recommendation_score': round(float(row['predicted_rating']) * 20, 1) # Convert 5 scale to 100
                })
                
            return recommendations
            
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            return []
