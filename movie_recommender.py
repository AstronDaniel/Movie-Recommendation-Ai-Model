"""
Movie Recommendation System
Implements content-based and collaborative filtering approaches
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
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
        self._initialize_movies()
        self._load_posters()
        
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
        
        return random_movies.to_dict('records')
