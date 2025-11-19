"""
Streamlit App for Movie Recommendation System - Netflix Style
"""

import streamlit as st
import pandas as pd
from pathlib import Path
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
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()

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

# Helper function to create movie card HTML
def create_movie_card(movie, index=None, score=None, poster_url=None, recommender=None):
    """Create a Netflix-style movie card with hover description"""
    title = movie['title']
    genre = movie['genre']
    year = movie['year']
    rating = movie['rating']
    director = movie.get('director', 'N/A')
    
    # Get description, with fallback to recommender
    description = get_movie_description(movie, recommender)
    
    # Truncate description for display
    desc_short = (description[:120] + '...') if len(description) > 120 and description != 'No description available.' else description
    
    # Get poster URL or use placeholder
    if poster_url:
        poster_html = f'<img src="{poster_url}" alt="{title}" class="poster-image" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">'
        poster_fallback = f'<div class="poster-fallback" style="display: none;">{title[0] if title else "M"}</div>'
    else:
        # Get first letter for poster placeholder
        poster_letter = title[0] if title else 'M'
        poster_html = ''
        poster_fallback = f'<div class="poster-fallback">{poster_letter}</div>'
    
    score_html = ""
    if score is not None:
        score_html = f'<div class="rating-badge" style="margin-top: 0.5rem;">{score}% Match</div>'
    
    # Escape HTML special characters
    title_escaped = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    genre_escaped = genre.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') if genre else 'Movie'
    desc_escaped = desc_short.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    
    # Create description overlay
    description_overlay = f'<div class="movie-description-overlay"><div class="description-text">{desc_escaped}</div></div>'
    
    return f'<div class="movie-card"><div class="movie-poster">{poster_html}{poster_fallback}{description_overlay}</div><div class="movie-info"><div class="movie-title">{title_escaped}</div><div class="movie-meta"><div>{year} • {genre_escaped.split()[0] if genre_escaped else "Movie"}</div><div class="movie-rating">Rating: {rating}/10</div>{score_html}</div></div></div>'

# Helper function to create horizontal scrolling row
def create_movie_row(title, movies, scores=None, recommender=None):
    """Create a horizontal scrolling row of movies"""
    movies_html = ""
    for i, movie in enumerate(movies):
        score = scores[i] if scores and i < len(scores) else None
        # Get poster URL from recommender if available
        poster_url = None
        if recommender and hasattr(recommender, 'get_poster_url'):
            try:
                poster_url = recommender.get_poster_url(movie.get('title', ''))
            except AttributeError:
                # If method doesn't exist (cached old version), skip poster
                poster_url = None
        movies_html += create_movie_card(movie, i, score, poster_url, recommender)
    
    return f'<div class="movie-row"><div class="row-title">{title}</div><div class="movie-scroll-container">{movies_html}</div></div>'

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
    
    # Display movies
    if search_query or search_button:
        movies = recommender.search_movies(search_query)
        if not movies.empty:
            movies_list = movies.to_dict('records')
            st.markdown(create_movie_row(f"Search Results ({len(movies)} movies)", movies_list, recommender=recommender), unsafe_allow_html=True)
        else:
            st.info("No movies found. Try a different search term.")
    else:
        # Show all movies in horizontal scrolling rows
        all_movies = recommender.get_all_movies()
        movies_list = all_movies.to_dict('records')
        
        # Split into rows of 10 movies each
        row_size = 10
        for i in range(0, len(movies_list), row_size):
            row_movies = movies_list[i:i+row_size]
            row_title = f"All Movies" if i == 0 else f"All Movies (continued)"
            st.markdown(create_movie_row(row_title, row_movies, recommender=recommender), unsafe_allow_html=True)
            
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
                    st.markdown(create_movie_row(f"Movies similar to {selected_movie}", recommendations, scores, recommender=recommender), unsafe_allow_html=True)
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
                        st.markdown(create_movie_row("Personalized Recommendations For You", recommendations, scores, recommender=recommender), unsafe_allow_html=True)
    
    else:  # Top Rated
        st.markdown("### Popular and highly-rated movies")
        st.markdown("Browse the most popular and highest-rated movies in our collection.")
        
        n_recommendations = st.slider("Number of movies to show", 3, 10, 5, key="top_rated_count")
        
        if st.button("Show Top Rated", key="top_rated", use_container_width=True):
            all_movies = recommender.get_all_movies()
            top_movies = all_movies.nlargest(n_recommendations, 'rating')
            top_movies_list = top_movies.to_dict('records')
            
            st.markdown(create_movie_row(f"Top {n_recommendations} Rated Movies", top_movies_list, recommender=recommender), unsafe_allow_html=True)

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
    st.markdown(create_movie_row("Selected Movie", [movie_dict], recommender=recommender), unsafe_allow_html=True)
    
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
        
        # Display in horizontal scrolling row
        st.markdown(create_movie_row("Your Rated Movies", rated_movies_list, recommender=recommender), unsafe_allow_html=True)
        
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
