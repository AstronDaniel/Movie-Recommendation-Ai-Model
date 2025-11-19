"""
Streamlit App for Movie Recommendation System - Netflix Style
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from movie_recommender import MovieRecommender

# Page configuration
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load external CSS
def load_css():
    """Load external CSS file"""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure assets/styles.css exists.")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

# Load CSS
load_css()

# Initialize the recommender system
@st.cache_resource
def get_recommender_system():
    return MovieRecommender()

recommender = get_recommender_system()

# Clear cache button in sidebar (for development) - show warning if method missing
if not hasattr(recommender, 'get_poster_url'):
    st.sidebar.warning("⚠️ Please restart Streamlit to load updated code, or click below to clear cache")
    if st.sidebar.button("🔄 Clear Cache & Reload", help="Clear Streamlit cache to reload updated code"):
        st.cache_resource.clear()
        st.rerun()

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'user_001'
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'page' not in st.session_state:
    st.session_state.page = 1

# Header
st.markdown("""
    <div class="netflix-header">
        <div class="netflix-logo" style="color: #e50914;">CINEMATCH AI</div>
        <div style="color: #ffffff; font-size: 1.1rem; font-weight: 500; letter-spacing: 0.5px;">Smart Movie Recommendations</div>
    </div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Discover Your Next Favorite Movie</div>
        <div class="hero-subtitle">AI-powered recommendations tailored to your taste. Search for any movie or explore our curated popular selections.</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This AI-powered movie recommendation system uses advanced machine learning techniques:
    
    **Content-Based Filtering**  
    Perfect for new users. Recommends movies similar to ones you like based on genre, director, and plot.
    
    **Collaborative Filtering**  
    For users with rating history. Finds movies liked by users with similar tastes.
    
    **Hybrid Approach**  
    Combines both methods for the best recommendations.
    """)
    
    st.divider()
    
    st.markdown("### User Profile")
    user_name = st.text_input("Enter your name", value="Movie Lover", help="Your name is used to personalize your experience")
    
    if st.session_state.user_ratings:
        st.metric("Movies Rated", len(st.session_state.user_ratings))
        if st.button("Clear Rating History", use_container_width=True):
            st.session_state.user_ratings = {}
            st.rerun()
    else:
        st.info("Start rating movies to get personalized recommendations!")

# Helper function to get description from recommender if missing
def get_movie_description(movie, recommender=None):
    """Get movie description, fetching from recommender if missing"""
    # First try to get from movie dict
    description = movie.get('description', '')
    
    # Check if description is actually missing or just the default message
    # If it's missing or empty, fetch from recommender's full dataset
    if (not description or 
        description == 'No description available.' or 
        description.strip() == '') and recommender:
        try:
            title = movie.get('title', '')
            if title and hasattr(recommender, 'movies_df') and recommender.movies_df is not None:
                movie_data = recommender.movies_df[recommender.movies_df['title'] == title]
                if not movie_data.empty and 'description' in movie_data.columns:
                    fetched_desc = movie_data.iloc[0].get('description', '')
                    if fetched_desc and fetched_desc.strip():
                        description = fetched_desc
        except Exception as e:
            # Silently fail and use default
            pass
    
    # Return description or default
    return description if description and description.strip() else 'No description available.'

# Movie Details Modal
@st.dialog("Movie Details")
def show_movie_details(movie):
    st.markdown(f"## {movie['title']}")
    
    # Create tabs for Overview and Trailer
    tab_overview, tab_trailer = st.tabs(["Overview", "Trailer"])
    
    with tab_overview:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            poster_url = movie.get('poster_path') or movie.get('poster_url')
            
            # Fix for raw TMDB paths if they somehow sneak in
            if poster_url and isinstance(poster_url, str) and poster_url.startswith('/'):
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_url}"
                
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown(f"<div style='height: 300px; background: #333; display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;'>{movie['title'][0]}</div>", unsafe_allow_html=True)
                
        with col2:
            st.markdown(f"""
            <div class="modal-meta">
                <span>{movie.get('year', 'N/A')}</span>
                <span>{movie.get('genre', 'Unknown')}</span>
                <span style="color: #46d369">Match: {movie.get('rating', 0)}/10</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"### Overview")
            st.markdown(f"<div class='modal-overview'>{movie.get('description', 'No description available.')}</div>", unsafe_allow_html=True)
            
            # Rating in modal
            current_rating = st.session_state.user_ratings.get(movie['title'], 0.0)
            new_rating = st.slider("Rate this movie", 0.0, 5.0, float(current_rating), 0.5, key=f"modal_rate_{movie['title']}")
            
            if st.button("Save Rating", key=f"save_rate_{movie['title']}"):
                st.session_state.user_ratings[movie['title']] = new_rating
                st.success("Rating saved!")
                st.rerun()

    with tab_trailer:
        # Fetch and display trailer
        movie_id = movie.get('movie_id') or movie.get('id')
        if movie_id:
            try:
                with st.spinner("Loading trailer..."):
                    videos = recommender.movie_api.get_movie_videos(movie_id)
                    # Filter for YouTube trailers
                    trailers = [v for v in videos if v.get('site') == 'YouTube' and v.get('type') == 'Trailer']
                    
                    if trailers:
                        # Get the first trailer
                        trailer_key = trailers[0].get('key')
                        if trailer_key:
                            st.video(f"https://www.youtube.com/watch?v={trailer_key}")
                        else:
                            st.info("Trailer video unavailable.")
                    else:
                        st.info("No trailer available for this movie.")
            except Exception as e:
                st.error(f"Could not load trailer. Error: {str(e)}")
        else:
            st.warning("Cannot load trailer: Movie ID missing.")

# Helper function to render a grid of movies
def render_movie_grid(movies, recommender=None, key_prefix="grid"):
    """Render movies in a responsive grid"""
    if not movies:
        st.warning("No movies to display.")
        return

    # CSS for grid container
    st.markdown('<div class="movie-grid-container">', unsafe_allow_html=True)
    
    # Calculate rows
    cols_per_row = 5
    rows = [movies[i:i + cols_per_row] for i in range(0, len(movies), cols_per_row)]
    
    for row_idx, row_movies in enumerate(rows):
        cols = st.columns(cols_per_row)
        for col_idx, movie in enumerate(row_movies):
            with cols[col_idx]:
                # Get poster
                poster_url = movie.get('poster_path')
                
                # Fix for raw TMDB paths
                if poster_url and isinstance(poster_url, str) and poster_url.startswith('/'):
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_url}"
                
                if not poster_url and recommender:
                     poster_url = recommender.get_poster_url(movie['title'])
                
                # Update movie dict with poster for modal
                movie['poster_url'] = poster_url
                
                # Render Card
                with st.container(border=True):
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown(f"<div style='height: 200px; background: #333; display: flex; align-items: center; justify-content: center; color: white;'>{movie['title'][0]}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"<div class='movie-card-title'>{movie['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='movie-card-meta'><span>{movie.get('year', '')}</span><span class='movie-card-rating'>★ {movie.get('rating', 0)}</span></div>", unsafe_allow_html=True)
                    
                    if st.button("Details", key=f"{key_prefix}_btn_{row_idx}_{col_idx}_{movie['title']}"):
                        show_movie_details(movie)
                        
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Browse", 
    "Recommendations", 
    "Rate Movies",
    "My List"
])

with tab1:
    st.markdown("### Browse Movie Collection")
    st.markdown("Search and explore our curated collection of movies")
    
    # Search functionality
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "Search for movies", 
            placeholder="Enter movie title, genre, or director...", 
            key="browse_search",
            help="Search by title, genre, or director name"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        search_button = st.button("Search", key="search_btn", use_container_width=True)
    
    # Advanced Filters & Random
    with st.expander("Advanced Filters & Discovery", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        # Get data for filters
        all_movies_df = recommender.get_all_movies()
        min_year = int(all_movies_df['year'].replace('', 0).astype(float).min()) if not all_movies_df.empty else 1900
        max_year = int(all_movies_df['year'].replace('', 2025).astype(float).max()) if not all_movies_df.empty else 2025
        if min_year == 0: min_year = 1900 # Handle case where year might be 0
        
        with f_col1:
            available_genres = recommender.get_genres()
            selected_genres = st.multiselect("Filter by Genre", available_genres, key="filter_genres")
            
        with f_col2:
            selected_years = st.slider("Release Year", min_year, max_year, (min_year, max_year), key="filter_years")
            
        with f_col3:
            selected_ratings = st.slider("Minimum Rating", 0.0, 10.0, (0.0, 10.0), step=0.5, key="filter_ratings")
            
        col_filter, col_random = st.columns([1, 1])
        with col_filter:
            apply_filters = st.button("Apply Filters", key="apply_filters_btn", use_container_width=True)
        with col_random:
            random_btn = st.button("🎲 Surprise Me!", key="random_btn", use_container_width=True, help="Get 10 random movie recommendations")
    
    # Logic for what to display
    if random_btn:
        st.session_state.view_mode = 'random'
    elif search_query or search_button:
        st.session_state.view_mode = 'search'
    elif apply_filters:
        st.session_state.view_mode = 'filtered'
    elif 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'popular'

    # Display based on view mode
    if st.session_state.view_mode == 'random':
        st.markdown("### 🎲 Random Picks For You")
        if st.button("← Back to Popular", key="back_random"):
            st.session_state.view_mode = 'popular'
            st.rerun()
            
        random_movies = recommender.get_random_movies(10)
        render_movie_grid(random_movies, recommender=recommender, key_prefix="random")

    elif st.session_state.view_mode == 'filtered':
        st.markdown("### Filtered Results")
        if st.button("Clear Filters", key="clear_filters"):
            st.session_state.view_mode = 'popular'
            st.rerun()
            
        with st.spinner("Filtering and fetching more movies from TMDB..."):
            filtered_df = recommender.filter_movies(
                genres=selected_genres,
                year_range=selected_years,
                rating_range=selected_ratings,
                fetch_from_api=True
            )
        
        if not filtered_df.empty:
            st.markdown(f"Found {len(filtered_df)} movies matching your criteria")
            render_movie_grid(filtered_df.to_dict('records'), recommender=recommender, key_prefix="filtered")
        else:
            st.info("No movies match your filters. Try adjusting your criteria.")

    elif st.session_state.view_mode == 'search':
        movies = recommender.search_movies(search_query)
        if not movies.empty:
            movies_list = movies.to_dict('records')
            st.markdown(f"### Search Results ({len(movies)} movies)")
            render_movie_grid(movies_list, recommender=recommender, key_prefix="search")
        else:
            st.info("No movies found. Try a different search term.")
            if st.button("Clear Search", key="clear_search"):
                st.session_state.view_mode = 'popular'
                st.rerun()

    else: # popular
        # Show all movies in grid
        st.markdown("### Popular Movies")
        all_movies = recommender.get_all_movies()
        movies_list = all_movies.to_dict('records')
        
        # Render grid
        render_movie_grid(movies_list, recommender=recommender, key_prefix="popular")
            
        # Load More button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load More Popular Movies", key="load_more_btn", use_container_width=True):
            with st.spinner("Fetching more movies from TMDB..."):
                st.session_state.page += 1
                new_movies = recommender.load_more_popular_movies(st.session_state.page)
                if not new_movies.empty:
                    st.success(f"Loaded {len(new_movies)} more movies!")
                    st.rerun()
                else:
                    st.warning("No more movies found.")

with tab2:
    # Choose recommendation type
    rec_type = st.radio(
        "Select recommendation type:",
        ["Similar Movies", "Personalized", "Top Rated"],
        horizontal=True
    )
    
    if rec_type == "Similar Movies":
        st.markdown("### Find movies similar to ones you like")
        st.markdown("Select a movie you enjoyed and discover similar titles based on content similarity.")
        
        # Movie selection
        all_movies = recommender.get_all_movies()
        selected_movie = st.selectbox(
            "Select a movie you enjoyed:",
            options=all_movies['title'].tolist(),
            key="similar_movie_select",
            help="Choose a movie you liked to find similar recommendations"
        )
        
        n_recommendations = st.slider("Number of recommendations", 3, 10, 5, key="similar_count")
        
        if st.button("Get Recommendations", key="content_rec", use_container_width=True):
            with st.spinner("Finding similar movies..."):
                recommendations = recommender.get_content_based_recommendations(
                    selected_movie, 
                    n_recommendations
                )
                
                if recommendations:
                    scores = [rec.get('similarity_score', 0) for rec in recommendations]
                    st.markdown(f"### Movies similar to {selected_movie}")
                    render_movie_grid(recommendations, recommender=recommender, key_prefix="similar")
                else:
                    st.warning("No recommendations found.")
    
    elif rec_type == "Personalized":
        st.markdown("### Get recommendations based on your rating history")
        st.markdown("Receive personalized recommendations based on your movie preferences and rating patterns.")
        
        if len(st.session_state.user_ratings) == 0:
            st.info("You haven't rated any movies yet. Please rate some movies in the 'Rate Movies' tab to get personalized recommendations.")
        else:
            n_recommendations = st.slider("Number of recommendations", 3, 10, 5, key="personalized_count")
            
            if st.button("Get Personalized Recommendations", key="collab_rec", use_container_width=True):
                # Add ratings to recommender
                for movie, rating in st.session_state.user_ratings.items():
                    recommender.add_user_rating(st.session_state.user_id, movie, rating)
                
                with st.spinner("Analyzing your preferences..."):
                    recommendations = recommender.get_collaborative_recommendations(
                        st.session_state.user_id,
                        n_recommendations
                    )
                    
                    if recommendations:
                        scores = [rec.get('recommendation_score', 0) for rec in recommendations]
                        st.markdown("### Personalized Recommendations For You")
                        render_movie_grid(recommendations, recommender=recommender, key_prefix="personalized")
    
    else:  # Top Rated
        st.markdown("### Popular and highly-rated movies")
        st.markdown("Browse the most popular and highest-rated movies in our collection.")
        
        n_recommendations = st.slider("Number of movies to show", 3, 10, 5, key="top_rated_count")
        
        if st.button("Show Top Rated", key="top_rated", use_container_width=True):
            all_movies = recommender.get_all_movies()
            top_movies = all_movies.nlargest(n_recommendations, 'rating')
            top_movies_list = top_movies.to_dict('records')
            
            st.markdown(f"### Top {n_recommendations} Rated Movies")
            render_movie_grid(top_movies_list, recommender=recommender, key_prefix="top_rated")

with tab3:
    st.markdown("### Rate Movies")
    st.markdown("Rate movies to improve your personalized recommendations. Your ratings help us understand your preferences better.")
    
    # Movie selection for rating
    all_movies = recommender.get_all_movies()
    movie_to_rate = st.selectbox(
        "Select a movie to rate:",
        options=all_movies['title'].tolist(),
        key="rate_movie_select",
        help="Choose a movie you've watched to rate it"
    )
    
    # Display movie info
    movie_info = all_movies[all_movies['title'] == movie_to_rate].iloc[0]
    
    # Show movie card - ensure description is included
    movie_dict = movie_info.to_dict()
    if 'description' not in movie_dict:
        # Fallback: get description from the full movies_df if missing
        full_movie = recommender.movies_df[recommender.movies_df['title'] == movie_to_rate]
        if not full_movie.empty:
            movie_dict['description'] = full_movie.iloc[0].get('description', 'No description available.')
    
    st.markdown("### Selected Movie")
    render_movie_grid([movie_dict], recommender=recommender, key_prefix="rate_selected")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Genre:** {movie_info['genre']}  
        **Year:** {movie_info['year']}  
        **Director:** {movie_info['director']}  
        **IMDB Rating:** {movie_info['rating']}/10
        """)
    
    with col2:
        # Check if already rated
        current_rating = st.session_state.user_ratings.get(movie_to_rate, 3.0)
        
    # Rating slider
    user_rating = st.slider(
        "Your rating (0.5 - 5.0):",
        min_value=0.5,
        max_value=5.0,
        value=float(current_rating),
        step=0.5,
        key="rating_slider"
    )
    
    # Display rating value
    st.markdown(f"### Rating: {user_rating}/5.0")
    
    # Submit rating
    if st.button("Submit Rating", key="submit_rating", use_container_width=True):
        st.session_state.user_ratings[movie_to_rate] = user_rating
        st.success(f"Successfully rated **{movie_to_rate}** with {user_rating}/5.0!")

with tab4:
    st.markdown("### My List")
    
    if st.session_state.user_ratings:
        st.markdown(f"**You have rated {len(st.session_state.user_ratings)} movies**")
        
        # Get movie details for rated movies
        all_movies = recommender.get_all_movies()
        rated_movies_list = []
        
        for movie_title, rating in st.session_state.user_ratings.items():
            movie_data = all_movies[all_movies['title'] == movie_title]
            if not movie_data.empty:
                movie_dict = movie_data.iloc[0].to_dict()
                movie_dict['user_rating'] = rating
                rated_movies_list.append(movie_dict)
        
        # Sort by user rating
        rated_movies_list.sort(key=lambda x: x['user_rating'], reverse=True)
        
        # Display in grid
        st.markdown("### Your Rated Movies")
        render_movie_grid(rated_movies_list, recommender=recommender, key_prefix="my_list")
        
        # Statistics
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ratings", len(st.session_state.user_ratings))
        with col2:
            avg_rating = sum(st.session_state.user_ratings.values()) / len(st.session_state.user_ratings)
            st.metric("Average Rating", f"{avg_rating:.2f}/5.0")
        with col3:
            highest = max(st.session_state.user_ratings.values())
            st.metric("Highest Rating", f"{highest}/5.0")
    else:
        st.info("You haven't rated any movies yet. Go to the 'Rate Movies' tab to start rating!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Movie Recommendation AI System | Built with Streamlit & scikit-learn</p>
    <p>Supports both Cold Start users and experienced users with rating history</p>
</div>
""", unsafe_allow_html=True)
