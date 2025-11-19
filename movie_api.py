"""
Movie API Service
Fetches movie artwork and metadata from TMDB (The Movie Database) API
"""

import requests
import os
from typing import Optional, Dict, List, Tuple
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MovieAPI:
    """Service to fetch movie artwork and metadata from TMDB API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize MovieAPI
        
        Args:
            api_key: TMDB API key. If not provided, will try to get from environment variable TMDB_API_KEY
        """
        self.api_key = api_key or os.getenv('TMDB_API_KEY', '')
        self.base_url = 'https://api.themoviedb.org/3'
        self.image_base_url = 'https://image.tmdb.org/t/p/w500'  # w500 for medium size posters
        
        if not self.api_key:
            print("Warning: TMDB_API_KEY not set. Movie posters will not be available.")
            print("Get a free API key at: https://www.themoviedb.org/settings/api")
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Search for a movie by title
        
        Args:
            title: Movie title
            year: Optional release year for better matching
            
        Returns:
            Movie data dictionary or None if not found
        """
        if not self.api_key:
            return None
            
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': title,
                'language': 'en-US'
            }
            
            if year:
                params['year'] = year
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                # Return the first (most relevant) result
                return data['results'][0]
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie data for '{title}': {e}")
            return None
    
    @lru_cache(maxsize=100)
    def get_movie_poster_url(self, title: str, year: Optional[int] = None) -> Optional[str]:
        """
        Get movie poster URL
        
        Args:
            title: Movie title
            year: Optional release year
            
        Returns:
            Full URL to movie poster image or None
        """
        movie_data = self.search_movie(title, year)
        
        if movie_data and movie_data.get('poster_path'):
            return f"{self.image_base_url}{movie_data['poster_path']}"
        
        return None
    
    def get_movie_details(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get detailed movie information including poster, backdrop, overview, etc.
        
        Args:
            title: Movie title
            year: Optional release year
            
        Returns:
            Dictionary with movie details or None
        """
        movie_data = self.search_movie(title, year)
        
        if not movie_data:
            return None
        
        # Get full movie details using movie ID
        try:
            movie_id = movie_data.get('id')
            if not movie_id:
                return movie_data
            
            url = f"{self.base_url}/movie/{movie_id}"
            params = {
                'api_key': self.api_key,
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            full_data = response.json()
            
            # Return formatted movie details
            return {
                'title': full_data.get('title', title),
                'poster_url': f"{self.image_base_url}{full_data['poster_path']}" if full_data.get('poster_path') else None,
                'backdrop_url': f"https://image.tmdb.org/t/p/w1280{full_data['backdrop_path']}" if full_data.get('backdrop_path') else None,
                'overview': full_data.get('overview', ''),
                'release_date': full_data.get('release_date', ''),
                'rating': full_data.get('vote_average', 0),
                'genres': [g['name'] for g in full_data.get('genres', [])],
                'runtime': full_data.get('runtime', 0),
                'tagline': full_data.get('tagline', '')
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie details for '{title}': {e}")
            return movie_data
    
    def get_movie_videos(self, movie_id: int) -> list:
        """
        Get videos (trailers, teasers, etc.) for a movie
        
        Args:
            movie_id: TMDB Movie ID
            
        Returns:
            List of video dictionaries
        """
        if not self.api_key or not movie_id:
            return []
            
        try:
            url = f"{self.base_url}/movie/{movie_id}/videos"
            params = {
                'api_key': self.api_key,
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            print(f"Video data for movie ID {movie_id}: {data}")
            return data.get('results', [])
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching videos for movie ID {movie_id}: {e}")
            return []

    def get_poster_url_batch(self, movies: list) -> Dict[str, str]:
        """
        Get poster URLs for multiple movies (with caching)
        
        Args:
            movies: List of movie dictionaries with 'title' and optionally 'year' keys
            
        Returns:
            Dictionary mapping movie titles to poster URLs
        """
        poster_urls = {}
        
        for movie in movies:
            title = movie.get('title', '')
            year = movie.get('year')
            
            if title:
                poster_url = self.get_movie_poster_url(title, year)
                if poster_url:
                    poster_urls[title] = poster_url
        
        return poster_urls

    def _get_genre_map(self):
        """Fetch genre mapping from TMDB"""
        if not self.api_key:
            return {}
        try:
            url = f"{self.base_url}/genre/movie/list"
            params = {'api_key': self.api_key, 'language': 'en-US'}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                genres = response.json().get('genres', [])
                return {g['id']: g['name'] for g in genres}
        except:
            pass
        return {}

    def _format_movie_data(self, data: Dict, genre_map: Dict = None) -> Dict:
        """Format TMDB movie data into our app's format"""
        if genre_map is None:
            genre_map = self._get_genre_map()
            
        genre_ids = data.get('genre_ids', [])
        genres = [genre_map.get(gid, str(gid)) for gid in genre_ids]
        
        poster_path = data.get('poster_path')
        if poster_path and poster_path.startswith('/'):
            poster_path = f"{self.image_base_url}{poster_path}"
            
        backdrop_path = data.get('backdrop_path')
        if backdrop_path and backdrop_path.startswith('/'):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        return {
            'title': data.get('title', ''),
            'genre': ' '.join(genres), 
            'year': int(data.get('release_date', '0000')[:4]) if data.get('release_date') else 0,
            'rating': data.get('vote_average', 0),
            'director': 'Unknown', 
            'description': data.get('overview', ''),
            'poster_path': poster_path,
            'backdrop_path': backdrop_path,
            'id': data.get('id'),
            'movie_id': data.get('id')
        }

    def search_movies(self, query: str) -> list:
        """
        Search for movies by title returning a list
        """
        if not self.api_key:
            return []
            
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': query,
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            results = []
            if data.get('results'):
                genre_map = self._get_genre_map()
                for item in data['results']:
                    results.append(self._format_movie_data(item, genre_map))
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching movies for '{query}': {e}")
            return []

    def get_popular_movies(self, page: int = 1) -> list:
        """Get popular movies from TMDB"""
        if not self.api_key:
            return []
            
        try:
            url = f"{self.base_url}/movie/popular"
            params = {
                'api_key': self.api_key,
                'language': 'en-US',
                'page': page
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            results = []
            if data.get('results'):
                genre_map = self._get_genre_map()
                for item in data['results']:
                    results.append(self._format_movie_data(item, genre_map))
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching popular movies: {e}")
            return []

    def get_top_rated_movies(self, page: int = 1) -> list:
        """Get top rated movies from TMDB"""
        if not self.api_key:
            return []
            
        try:
            url = f"{self.base_url}/movie/top_rated"
            params = {
                'api_key': self.api_key,
                'language': 'en-US',
                'page': page
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            results = []
            if data.get('results'):
                genre_map = self._get_genre_map()
                for item in data['results']:
                    results.append(self._format_movie_data(item, genre_map))
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching top rated movies: {e}")
            return []

    def discover_movies(
        self, 
        genre_ids: List[int] = None, 
        year_range: Tuple[int, int] = None,
        rating_min: float = None
    ) -> list:
        """
        Discover movies using TMDB API with filters
        
        Args:
            genre_ids: List of genre IDs to filter by
            year_range: Tuple of (start_year, end_year) to filter by release year
            rating_min: Minimum average rating to filter by
            
        Returns:
            List of discovered movies
        """
        if not self.api_key:
            return []
            
        try:
            url = f"{self.base_url}/discover/movie"
            params = {
                'api_key': self.api_key,
                'language': 'en-US',
                'sort_by': 'popularity.desc',
                'page': 1
            }
            
            if genre_ids:
                params['with_genres'] = ','.join(map(str, genre_ids))
                
            if year_range:
                params['primary_release_date.gte'] = f"{year_range[0]}-01-01"
                params['primary_release_date.lte'] = f"{year_range[1]}-12-31"
                
            if rating_min is not None:
                params['vote_average.gte'] = rating_min
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            results = []
            if data.get('results'):
                genre_map = self._get_genre_map()
                for item in data['results']:
                    results.append(self._format_movie_data(item, genre_map))
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error discovering movies: {e}")
            return []

    def get_genre_id_map(self) -> Dict[str, int]:
        """Get mapping of genre name to ID"""
        genre_map = self._get_genre_map()
        return {name: id for id, name in genre_map.items()}
