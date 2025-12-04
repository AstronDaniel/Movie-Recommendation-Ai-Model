import os
import pickle
import traceback
import re
import requests
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, 'models')

# API Keys from .env
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '')

# TMDB base URLs
TMDB_SEARCH_URL = 'https://api.themoviedb.org/3/search/movie'
TMDB_IMG_BASE = 'https://image.tmdb.org/t/p/w500'

app = Flask(__name__, static_folder='static', template_folder='templates')

# Globals to hold loaded models/data
cf_model = None
similarity_df = None
movies_df = None
prep_data = None
hybrid_config = None


def try_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def load_models():
    global cf_model, similarity_df, movies_df, prep_data, hybrid_config
    print("Loading models...")
    # Try common filenames
    files = {
        'cf_model.pkl': 'cf_model',
        'similarity_matrix.pkl': 'similarity_df',
        'movies_data.pkl': 'movies_df',
        'movies_processed.pkl': 'movies_df',
        'preprocessing_data.pkl': 'prep_data',
        'hybrid_config.pkl': 'hybrid_config'
    }

    for fname, varname in files.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            try:
                print(f"Attempting to load {fname}...")
                obj = try_load(fpath)
                if obj is not None:
                    globals()[varname] = obj
                    print(f'Loaded {fname} -> {varname}')
                else:
                    print(f'Failed to load {fname} (returned None)')
            except Exception as e:
                print(f'Exception loading {fname}: {e}')
        else:
            print(f"File not found: {fpath}")

    # If movies_df is a dict or DataFrame-like, coerce to DataFrame
    if movies_df is not None and not isinstance(movies_df, pd.DataFrame):
        try:
            movies_df = pd.DataFrame(movies_df)
            print("Converted movies_df to DataFrame")
        except Exception as e:
            print(f"Failed to convert movies_df: {e}")
            movies_df = None
    
    if movies_df is None:
        print("WARNING: movies_df is None after loading!")
    else:
        print(f"movies_df loaded with shape: {movies_df.shape}")


# ============================================
# TMDB API INTEGRATION
# ============================================

def clean_title_for_search(title):
    """Remove year suffix and clean title for TMDB search."""
    # Remove trailing year like "Toy Story (1995)"
    cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', str(title))
    return cleaned.strip()


@lru_cache(maxsize=500)
def fetch_tmdb_poster(title, year=None):
    """
    Fetch movie poster URL from TMDB API.
    Uses LRU cache to avoid repeated API calls.
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        clean_title = clean_title_for_search(title)
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_title,
            'include_adult': 'false'
        }
        if year:
            params['year'] = str(year)
        
        resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"{TMDB_IMG_BASE}{poster_path}"
    except Exception as e:
        app.logger.warning(f'TMDB fetch failed for "{title}": {e}')
    
    return None


@lru_cache(maxsize=500)
def fetch_tmdb_details(title, year=None):
    """
    Fetch movie details (poster, overview, rating) from TMDB.
    """
    if not TMDB_API_KEY:
        return {}
    
    try:
        clean_title = clean_title_for_search(title)
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_title,
            'include_adult': 'false'
        }
        if year:
            params['year'] = str(year)
        
        resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            if results:
                movie = results[0]
                poster_path = movie.get('poster_path')
                return {
                    'poster': f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None,
                    'overview': movie.get('overview', '')[:200],
                    'tmdb_rating': movie.get('vote_average'),
                    'release_date': movie.get('release_date', '')
                }
    except Exception as e:
        app.logger.warning(f'TMDB details failed for "{title}": {e}')
    
    return {}


def enrich_movie(movie_dict):
    """
    Add poster and details to a movie dict.
    """
    title = movie_dict.get('title', '')
    year = movie_dict.get('year', '')
    
    # Try to get poster
    details = fetch_tmdb_details(title, year if year else None)
    movie_dict['poster'] = details.get('poster')
    movie_dict['overview'] = details.get('overview', '')
    movie_dict['tmdb_rating'] = details.get('tmdb_rating')
    
    return movie_dict


# Replaced by the block above


def recommend_for_user(user_id, n=10):
    # Try collaborative predictions first (if cf_model available)
    recs = []
    try:
        if cf_model is not None and movies_df is not None:
            # Predict rating for every movie and sort
            scores = []
            movie_ids = movies_df['movie_id'].unique()
            for mid in movie_ids:
                try:
                    # Surprise SVD models have .predict(uid, iid)
                    pred = cf_model.predict(str(user_id), str(mid)) if hasattr(cf_model, 'predict') else cf_model.predict(user_id, mid)
                    est = getattr(pred, 'est', pred)
                except Exception:
                    # fallback if predict signature differs
                    try:
                        pred = cf_model.predict(user_id, mid)
                        est = getattr(pred, 'est', pred)
                    except Exception:
                        est = 0
                scores.append((mid, float(est)))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top = scores[:n]
            for mid, score in top:
                row = movies_df[movies_df['movie_id'] == int(mid)]
                if not row.empty:
                    r = row.iloc[0]
                    movie = {
                        'movie_id': int(mid),
                        'title': r.get('title', ''),
                        'genres': r.get('genres', ''),
                        'year': r.get('year', ''),
                        'score': score
                    }
                    movie = enrich_movie(movie)
                    recs.append(movie)
            if recs:
                return recs

        # If CF not available, try content-based similarity
        if similarity_df is not None and movies_df is not None:
            # Recommend top movies by average similarity (simple heuristic)
            sim_sums = similarity_df.sum(axis=1)
            best = sim_sums.sort_values(ascending=False).head(n).index.tolist()
            for mid in best:
                row = movies_df[movies_df['movie_id'] == int(mid)]
                if not row.empty:
                    r = row.iloc[0]
                    movie = {
                        'movie_id': int(mid),
                        'title': r.get('title', ''),
                        'genres': r.get('genres', ''),
                        'year': r.get('year', ''),
                        'score': None
                    }
                    movie = enrich_movie(movie)
                    recs.append(movie)
            if recs:
                return recs

    except Exception:
        app.logger.warning('Recommend error: ' + traceback.format_exc())

    # Fallback: top-rated movies from movies_df
    try:
        if movies_df is not None and 'avg_rating' in movies_df.columns:
            top_df = movies_df.sort_values('avg_rating', ascending=False).head(n)
        else:
            top_df = movies_df.head(n) if movies_df is not None else []
        for _, r in top_df.iterrows():
            movie = {
                'movie_id': int(r.get('movie_id', 0)),
                'title': r.get('title', 'Unknown'),
                'genres': r.get('genres', ''),
                'year': r.get('year', ''),
                'score': r.get('avg_rating', None)
            }
            movie = enrich_movie(movie)
            recs.append(movie)
    except Exception:
        pass

    return recs



@lru_cache(maxsize=1)
def fetch_now_playing():
    """
    Fetch 'Now Playing' movies from TMDB.
    """
    if not TMDB_API_KEY:
        return []
    
    try:
        url = f"https://api.themoviedb.org/3/movie/now_playing"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'page': 1
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            movies = []
            for m in results[:12]:  # Top 12
                movies.append({
                    'movie_id': m.get('id'), # Note: TMDB ID, not our internal ID
                    'title': m.get('title'),
                    'poster': f"{TMDB_IMG_BASE}{m.get('poster_path')}" if m.get('poster_path') else None,
                    'release_date': m.get('release_date'),
                    'overview': m.get('overview', '')[:150] + '...' if m.get('overview') else ''
                })
            return movies
    except Exception as e:
        app.logger.warning(f'Now Playing fetch failed: {e}')
    
    return []


# Routes replaced below




def get_top_rated_movies(n=12):
    """
    Get top rated movies from our internal dataset.
    This serves as the 'AI Recommended' list for new users (Cold Start).
    """
    if movies_df is not None and not movies_df.empty:
        try:
            # Sort by rating (and maybe vote count if we had it, but we'll stick to simple avg_rating)
            # We assume 'avg_rating' exists from our training/preprocessing
            if 'avg_rating' in movies_df.columns:
                top_df = movies_df.sort_values('avg_rating', ascending=False).head(n)
            else:
                # Fallback if no rating column
                top_df = movies_df.head(n)
                
            results = []
            for _, r in top_df.iterrows():
                movie = {
                    'movie_id': int(r.get('movie_id', 0)),
                    'title': r.get('title', 'Unknown Title'),
                    'genres': r.get('genres', ''),
                    'year': r.get('year', ''),
                    'rating': r.get('avg_rating', None)
                }
                movie = enrich_movie(movie)
                results.append(movie)
            return results
        except Exception as e:
            app.logger.error(f"Top rated error: {e}")
    return []


@app.route('/')
def index():
    # 1. Get "Trending Now" (Now Playing) from TMDB
    trending = fetch_now_playing()
    
    # 2. Get "Top Picks for You" (Top Rated from DB) - AI Recommendation for New Users
    top_picks = get_top_rated_movies(12)
    
    # 3. Get "Featured Collection" (Random sample from our DB)
    featured = []
    if movies_df is not None and not movies_df.empty:
        try:
            # Randomize every time
            sample_df = movies_df.sample(n=12)
            for _, r in sample_df.iterrows():
                movie = {
                    'movie_id': int(r.get('movie_id', 0)),
                    'title': r.get('title', 'Unknown Title'),
                    'genres': r.get('genres', ''),
                    'year': r.get('year', '')
                }
                movie = enrich_movie(movie)
                featured.append(movie)
        except Exception:
            pass

    print(f"DEBUG: Index Route - Trending: {len(trending)}, Top Picks: {len(top_picks)}, Featured: {len(featured)}")
    return render_template('index.html', trending=trending, top_picks=top_picks, featured=featured)


@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return render_template('index.html')
    
    results = []
    
    # 1. Search in TMDB (Best for visuals and new movies)
    if TMDB_API_KEY:
        try:
            params = {'api_key': TMDB_API_KEY, 'query': query}
            resp = requests.get(TMDB_SEARCH_URL, params=params)
            if resp.status_code == 200:
                data = resp.json().get('results', [])
                for m in data:
                    if m.get('poster_path'): # Only show if it has a poster
                        results.append({
                            'movie_id': m.get('id'), # TMDB ID
                            'title': m.get('title'),
                            'poster': f"{TMDB_IMG_BASE}{m.get('poster_path')}",
                            'year': m.get('release_date', '')[:4],
                            'is_tmdb': True
                        })
        except Exception as e:
            app.logger.error(f"Search error: {e}")

    return render_template('index.html', search_results=results, search_query=query)


@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    # Determine if this is a TMDB ID or Internal ID
    # For simplicity in this hybrid app, we treat the ID passed in URL as TMDB ID if possible,
    # or we try to map it.
    
    # However, our `movies_df` has internal IDs.
    # If the user clicked from "Featured" (Internal ID), we need to handle that.
    # If they clicked from "Trending" or "Search" (TMDB ID), we handle that.
    
    # Strategy:
    # 1. Fetch details from TMDB using the ID (assuming it's a TMDB ID first).
    # 2. If that fails, or if we know it's internal, we might need a mapping.
    # BUT: `enrich_movie` uses title search to get TMDB data.
    # Let's assume for the "Details Page" we primarily rely on TMDB data for the visuals/trailer.
    
    # If the ID comes from our internal DB, we might not have the TMDB ID directly unless we search.
    # To make this robust:
    # - Pass `?type=internal` if it's from our DB.
    # - Else assume TMDB ID.
    
    is_internal = request.args.get('type') == 'internal'
    
    movie_data = {}
    recommendations = []
    
    if is_internal:
        # It's an internal ID. Get title from DB, then search TMDB for details.
        if movies_df is not None:
            row = movies_df[movies_df['movie_id'] == movie_id]
            if not row.empty:
                title = row.iloc[0]['title']
                # Search TMDB for this title to get the REAL TMDB ID for trailers/etc
                tmdb_search = fetch_tmdb_details(title) # This returns dict, not ID.
                # We need the ID for trailer.
                # Let's do a quick search to get ID
                try:
                    params = {'api_key': TMDB_API_KEY, 'query': clean_title_for_search(title)}
                    resp = requests.get(TMDB_SEARCH_URL, params=params)
                    if resp.status_code == 200 and resp.json()['results']:
                        tmdb_id = resp.json()['results'][0]['id']
                        return redirect(f"/movie/{tmdb_id}") # Redirect to TMDB ID version
                except:
                    pass
    
    # Assume movie_id is TMDB ID
    tmdb_id = movie_id
    
    # 1. Fetch Details & Trailer
    if TMDB_API_KEY:
        try:
            # Details
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            params = {'api_key': TMDB_API_KEY, 'append_to_response': 'videos,credits'}
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                movie_data = {
                    'id': data.get('id'),
                    'title': data.get('title'),
                    'overview': data.get('overview'),
                    'poster': f"{TMDB_IMG_BASE}{data.get('poster_path')}",
                    'backdrop': f"https://image.tmdb.org/t/p/original{data.get('backdrop_path')}" if data.get('backdrop_path') else None,
                    'year': data.get('release_date', '')[:4],
                    'rating': data.get('vote_average'),
                    'runtime': data.get('runtime'),
                    'genres': [g['name'] for g in data.get('genres', [])],
                    'cast': [c['name'] for c in data.get('credits', {}).get('cast', [])[:5]],
                    'trailer': None
                }
                
                # Find trailer
                videos = data.get('videos', {}).get('results', [])
                for v in videos:
                    if v['site'] == 'YouTube' and v['type'] == 'Trailer':
                        movie_data['trailer'] = v['key']
                        break
        except Exception as e:
            app.logger.error(f"Details error: {e}")

    # 2. Get Recommendations
    # Strategy: Use TMDB 'similar' endpoint for best results with new movies.
    # OR use our model if we can map it.
    # For this "Netflix" experience, TMDB similar is often higher quality for display.
    # BUT we want to show off our model?
    # Let's try to find the movie in our DB by title.
    
    model_recs = []
    if movies_df is not None and movie_data.get('title'):
        # Try to find by title
        # This is fuzzy and might fail, but let's try
        # Simple exact match (case insensitive)
        title_lower = movie_data['title'].lower()
        # Create a temp column for searching
        # This is slow, but okay for a demo
        # A better way is to have a lookup dict
        pass 

    # Fallback: Use TMDB Similar
    if TMDB_API_KEY:
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/similar"
            params = {'api_key': TMDB_API_KEY}
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                for m in results[:12]:
                    if m.get('poster_path'):
                        recommendations.append({
                            'movie_id': m.get('id'),
                            'title': m.get('title'),
                            'poster': f"{TMDB_IMG_BASE}{m.get('poster_path')}",
                            'is_tmdb': True
                        })
        except:
            pass

    return render_template('movie_details.html', movie=movie_data, recommendations=recommendations)


# Load models on startup
load_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
