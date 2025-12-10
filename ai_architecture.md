# CineMind AI Architecture

## Overview
CineMind AI is a hybrid movie recommendation system that combines **Collaborative Filtering (SVD)**, **Content-Based Filtering**, and **Real-Time API Data** to provide personalized and relevant movie suggestions.

## 1. The Hybrid Model
The core recommendation engine is built on a hybrid approach to solve the "Cold Start" problem and provide us with diverse results.

### A. Collaborative Filtering (SVD)
- **Algorithm**: Singular Value Decomposition (SVD) from the `surprise` library.
- **Training**: Trained on the MovieLens dataset (User-Movie ratings).
- **Function**: Predicts a user's rating for a movie they haven't seen yet.
- **Usage**: When a User ID is provided (Advanced Mode), this model ranks movies based on predicted preference.

### B. Content-Based Filtering
- **Algorithm**: Cosine Similarity on TF-IDF vectors.
- **Features**: Movie Genres, Keywords, Cast, and Director.
- **Training**: A similarity matrix is pre-computed during the training phase.
- **Usage**:
    - **"More Like This"**: When viewing a movie, we find the most similar movies in vector space.
    - **Fallback**: If the SVD model fails or for new users, we use content similarity to trending movies.

### C. Real-Time "Trending" (TMDB API)
- **Source**: The Movie Database (TMDB) API.
- **Usage**: Fetches "Now Playing" and "Trending" movies to ensure the platform isn't stuck in the past (since MovieLens data is historical).
- **Integration**: These movies are displayed in the "Trending Now" carousel.

## 2. Data Flow

### User Visit (Home Page)
1.  **Cold Start**: The system fetches "Now Playing" movies from TMDB.
2.  **AI Top Picks**: The system queries the internal database for the highest-rated movies (based on historical user ratings) to populate "Top Picks for You".
3.  **Featured**: A randomized sample of movies from the internal database is shown to encourage exploration.

### Movie Details & Recommendations
1.  **User Action**: User clicks on a movie (e.g., "Inception").
2.  **Data Fetch**:
    - If it's a **TMDB Movie**: We fetch details and use TMDB's `similar` endpoint (which uses their own massive user data).
    - If it's an **Internal Movie**: We look up the movie in our `similarity_df` (Content-Based) to find the top 10 most similar movies.
3.  **Display**: The UI shows the movie backdrop, trailer, and the list of recommended movies.

## 3. Technical Stack
- **Backend**: Flask (Python)
- **ML Libraries**: `scikit-learn`, `surprise`, `pandas`
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript
- **API**: TMDB (The Movie Database)

## 4. Future Improvements
- **User Accounts**: Allow users to rate movies to fine-tune the SVD model in real-time.
- **Deep Learning**: Implement a Neural Collaborative Filtering (NCF) model for better accuracy.
