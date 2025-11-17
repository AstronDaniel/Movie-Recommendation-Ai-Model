"""
Movie Recommendation System
Implements content-based and collaborative filtering approaches
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple


class MovieRecommender:
    """Movie recommendation system supporting both cold start and experienced users"""
    
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.user_ratings = {}
        self._initialize_movies()
        
    def _initialize_movies(self):
        """Initialize with a sample movie dataset"""
        # Sample movie dataset with genres, directors, and descriptions
        movies_data = {
            'movie_id': range(1, 21),
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Interstellar',
                'The Green Mile', 'Saving Private Ryan', 'The Prestige',
                'The Departed', 'Gladiator', 'The Lion King', 'Titanic',
                'Avatar', 'Avengers: Endgame', 'Joker'
            ],
            'genre': [
                'Drama Crime', 'Crime Drama', 'Action Crime Drama',
                'Crime Drama', 'Drama Romance', 'Action Sci-Fi Thriller',
                'Action Sci-Fi', 'Crime Drama', 'Crime Drama Thriller',
                'Adventure Drama Sci-Fi', 'Crime Drama Fantasy', 'Drama War',
                'Drama Mystery Sci-Fi', 'Crime Drama Thriller', 'Action Drama',
                'Animation Adventure Drama', 'Drama Romance', 'Action Adventure Fantasy',
                'Action Adventure Sci-Fi', 'Crime Drama Thriller'
            ],
            'director': [
                'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan',
                'Quentin Tarantino', 'Robert Zemeckis', 'Christopher Nolan',
                'Lana Wachowski', 'Martin Scorsese', 'Jonathan Demme',
                'Christopher Nolan', 'Frank Darabont', 'Steven Spielberg',
                'Christopher Nolan', 'Martin Scorsese', 'Ridley Scott',
                'Roger Allers', 'James Cameron', 'James Cameron',
                'Anthony Russo', 'Todd Phillips'
            ],
            'year': [
                1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 2014,
                1999, 1998, 2006, 2006, 2000, 1994, 1997, 2009, 2019, 2019
            ],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6,
                      8.6, 8.6, 8.5, 8.5, 8.5, 8.5, 7.9, 7.8, 8.4, 8.4],
            'description': [
                'Two imprisoned men bond over years finding redemption through acts of common decency',
                'The aging patriarch of an organized crime dynasty transfers control to his reluctant son',
                'Batman raises the stakes in his war on crime with the help of Lt. Gordon and Harvey Dent',
                'The lives of two mob hitmen a boxer and a pair of diner bandits intertwine',
                'The presidencies of Kennedy and Johnson through the perspective of an Alabama man',
                'A thief who steals corporate secrets through dream-sharing technology',
                'A computer hacker learns about the true nature of reality and his role in the war',
                'The story of Henry Hill and his life in the mob',
                'A young FBI cadet must receive help from an incarcerated cannibal killer',
                'A team of explorers travel through a wormhole in space',
                'The lives of guards on Death Row are affected by one of their charges',
                'Following the Normandy Landings a group sets out to find a paratrooper',
                'After a tragic accident two stage magicians engage in a battle to create the ultimate illusion',
                'An undercover cop and a mole in the police try to identify each other',
                'A former Roman General sets out to exact vengeance against the corrupt emperor',
                'Lion cub prince flees his kingdom only to learn the true meaning of responsibility',
                'A seventeen-year-old aristocrat falls in love with a kind but poor artist',
                'A paraplegic Marine dispatched to the moon Pandora on a unique mission',
                'After the devastating events the Avengers assemble once more to reverse Thanos actions',
                'In Gotham City mentally troubled comedian Arthur Fleck turns to a life of crime'
            ]
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        
        # Create content features by combining genre, director, and description
        self.movies_df['content'] = (
            self.movies_df['genre'] + ' ' + 
            self.movies_df['director'] + ' ' + 
            self.movies_df['description']
        )
        
        # Create TF-IDF matrix for content-based filtering
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def get_all_movies(self) -> pd.DataFrame:
        """Return all movies in the dataset"""
        return self.movies_df[['movie_id', 'title', 'genre', 'year', 'rating', 'director']]
    
    def search_movies(self, query: str) -> pd.DataFrame:
        """Search for movies by title"""
        if not query:
            return self.get_all_movies()
        
        mask = self.movies_df['title'].str.contains(query, case=False, na=False)
        return self.movies_df[mask][['movie_id', 'title', 'genre', 'year', 'rating', 'director']]
    
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
                'recommendation_score': round(score * 100, 2)
            })
        
        return recommendations
    
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
