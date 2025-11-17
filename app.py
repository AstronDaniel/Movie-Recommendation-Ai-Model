"""
Streamlit App for Movie Recommendation System
"""

import streamlit as st
import pandas as pd
from movie_recommender import MovieRecommender

# Page configuration
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-score {
        background-color: #FF4B4B;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the recommender system
@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'user_001'
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Header
st.title("🎬 Movie Recommendation AI System")
st.markdown("### Discover your next favorite movie with AI-powered recommendations")

# Sidebar
with st.sidebar:
    st.header("🎯 About")
    st.markdown("""
    This AI-powered movie recommendation system uses:
    - **Content-Based Filtering**: For new users (Cold Start)
    - **Collaborative Filtering**: For users with rating history
    - **Hybrid Approach**: Best of both worlds
    
    Get personalized movie recommendations based on your preferences!
    """)
    
    st.divider()
    
    st.header("👤 User Profile")
    user_name = st.text_input("Enter your name", value="Movie Lover")
    
    if st.session_state.user_ratings:
        st.metric("Movies Rated", len(st.session_state.user_ratings))
        if st.button("Clear Rating History"):
            st.session_state.user_ratings = {}
            st.rerun()

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Browse Movies", 
    "🎯 Get Recommendations", 
    "⭐ Rate Movies",
    "📊 My Ratings"
])

with tab1:
    st.header("Browse Movie Collection")
    
    # Search functionality
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔎 Search for movies", placeholder="Enter movie title...")
    with col2:
        search_button = st.button("Search", key="search_btn")
    
    # Display movies
    if search_query or search_button:
        movies = recommender.search_movies(search_query)
    else:
        movies = recommender.get_all_movies()
    
    if not movies.empty:
        st.markdown(f"**Found {len(movies)} movies**")
        
        # Display movies in a nice format
        for idx, row in movies.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"### {row['title']}")
                    st.markdown(f"**Genre:** {row['genre']}")
                    st.markdown(f"**Director:** {row['director']}")
                
                with col2:
                    st.markdown(f"**Year:** {row['year']}")
                    st.markdown(f"**Rating:** ⭐ {row['rating']}/10")
                
                with col3:
                    st.markdown("")  # Spacing
                
                st.divider()
    else:
        st.info("No movies found. Try a different search term.")

with tab2:
    st.header("Get Personalized Recommendations")
    
    # Choose recommendation type
    rec_type = st.radio(
        "Select recommendation type:",
        ["Content-Based (Similar Movies)", "Personalized (Based on Your Ratings)", "Top Rated"],
        horizontal=True
    )
    
    if rec_type == "Content-Based (Similar Movies)":
        st.markdown("#### Find movies similar to ones you like")
        
        # Movie selection
        all_movies = recommender.get_all_movies()
        selected_movie = st.selectbox(
            "Select a movie you enjoyed:",
            options=all_movies['title'].tolist()
        )
        
        n_recommendations = st.slider("Number of recommendations", 3, 10, 5)
        
        if st.button("Get Recommendations", key="content_rec"):
            with st.spinner("Finding similar movies..."):
                recommendations = recommender.get_content_based_recommendations(
                    selected_movie, 
                    n_recommendations
                )
                
                if recommendations:
                    st.success(f"🎬 Movies similar to **{selected_movie}**:")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="movie-card">
                                <h3>{i}. {rec['title']}</h3>
                                <p><strong>Genre:</strong> {rec['genre']} | 
                                <strong>Year:</strong> {rec['year']} | 
                                <strong>Rating:</strong> ⭐ {rec['rating']}/10</p>
                                <p><strong>Director:</strong> {rec['director']}</p>
                                <span class="recommendation-score">
                                    {rec['similarity_score']}% Match
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found.")
    
    elif rec_type == "Personalized (Based on Your Ratings)":
        st.markdown("#### Get recommendations based on your rating history")
        
        if len(st.session_state.user_ratings) == 0:
            st.info("⚠️ You haven't rated any movies yet. Please rate some movies in the 'Rate Movies' tab to get personalized recommendations.")
        else:
            n_recommendations = st.slider("Number of recommendations", 3, 10, 5)
            
            if st.button("Get Personalized Recommendations", key="collab_rec"):
                # Add ratings to recommender
                for movie, rating in st.session_state.user_ratings.items():
                    recommender.add_user_rating(st.session_state.user_id, movie, rating)
                
                with st.spinner("Analyzing your preferences..."):
                    recommendations = recommender.get_collaborative_recommendations(
                        st.session_state.user_id,
                        n_recommendations
                    )
                    
                    if recommendations:
                        st.success(f"🎬 Personalized recommendations for you:")
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.container():
                                score_key = 'recommendation_score' if 'recommendation_score' in rec else 'recommendation_reason'
                                score_value = rec.get('recommendation_score', rec.get('recommendation_reason', 'N/A'))
                                
                                st.markdown(f"""
                                <div class="movie-card">
                                    <h3>{i}. {rec['title']}</h3>
                                    <p><strong>Genre:</strong> {rec['genre']} | 
                                    <strong>Year:</strong> {rec['year']} | 
                                    <strong>Rating:</strong> ⭐ {rec['rating']}/10</p>
                                    <p><strong>Director:</strong> {rec['director']}</p>
                                    <span class="recommendation-score">
                                        Score: {score_value}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
    
    else:  # Top Rated
        st.markdown("#### Popular and highly-rated movies")
        
        n_recommendations = st.slider("Number of movies to show", 3, 10, 5)
        
        if st.button("Show Top Rated", key="top_rated"):
            all_movies = recommender.get_all_movies()
            top_movies = all_movies.nlargest(n_recommendations, 'rating')
            
            st.success(f"🏆 Top {n_recommendations} Rated Movies:")
            
            for i, (idx, movie) in enumerate(top_movies.iterrows(), 1):
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3>{i}. {movie['title']}</h3>
                        <p><strong>Genre:</strong> {movie['genre']} | 
                        <strong>Year:</strong> {movie['year']} | 
                        <strong>Rating:</strong> ⭐ {movie['rating']}/10</p>
                        <p><strong>Director:</strong> {movie['director']}</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab3:
    st.header("Rate Movies")
    st.markdown("Rate movies to get better personalized recommendations!")
    
    # Movie selection for rating
    all_movies = recommender.get_all_movies()
    movie_to_rate = st.selectbox(
        "Select a movie to rate:",
        options=all_movies['title'].tolist(),
        key="rate_movie_select"
    )
    
    # Display movie info
    movie_info = all_movies[all_movies['title'] == movie_to_rate].iloc[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Genre:** {movie_info['genre']}  
        **Year:** {movie_info['year']}  
        **Director:** {movie_info['director']}  
        **IMDB Rating:** ⭐ {movie_info['rating']}/10
        """)
    
    with col2:
        # Check if already rated
        current_rating = st.session_state.user_ratings.get(movie_to_rate, 3.0)
        
    # Rating slider
    user_rating = st.slider(
        "Your rating:",
        min_value=0.5,
        max_value=5.0,
        value=float(current_rating),
        step=0.5,
        key="rating_slider"
    )
    
    # Display star rating
    stars = "⭐" * int(user_rating) + "☆" * (5 - int(user_rating))
    st.markdown(f"### {stars} ({user_rating}/5.0)")
    
    # Submit rating
    if st.button("Submit Rating", key="submit_rating"):
        st.session_state.user_ratings[movie_to_rate] = user_rating
        st.success(f"✅ Rated **{movie_to_rate}** with {user_rating}/5.0 stars!")
        st.balloons()

with tab4:
    st.header("My Ratings")
    
    if st.session_state.user_ratings:
        st.markdown(f"**You have rated {len(st.session_state.user_ratings)} movies**")
        
        # Convert ratings to DataFrame for display
        ratings_df = pd.DataFrame([
            {
                'Movie': movie,
                'Your Rating': f"{'⭐' * int(rating)} ({rating}/5.0)",
                'Rating Value': rating
            }
            for movie, rating in st.session_state.user_ratings.items()
        ]).sort_values('Rating Value', ascending=False)
        
        # Display ratings
        for idx, row in ratings_df.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{row['Movie']}**")
            with col2:
                st.markdown(f"{row['Your Rating']}")
            st.divider()
        
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
<div style='text-align: center; color: #666;'>
    <p>🎬 Movie Recommendation AI System | Built with Streamlit & scikit-learn</p>
    <p>Supports both Cold Start users and experienced users with rating history</p>
</div>
""", unsafe_allow_html=True)
