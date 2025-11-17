# Movie Recommendation AI Model 🎬

![Image](https://videos.openai.com/az/vg-assets/task_01ka8hxk68e9s9yqj43r1zae86%2F1763371413_img_0.webp?se=2025-11-20T00%3A00%3A00Z&sp=r&sv=2024-08-04&sr=b&skoid=8ebb0df1-a278-4e2e-9c20-f2d373479b3a&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-11-17T01%3A48%3A57Z&ske=2025-11-24T01%3A53%3A57Z&sks=b&skv=2024-08-04&sig=Nn60F523T/oJJnfE8YB5RqgQAROQNRfPauJ8TT1TXEg%3D&ac=oaivgprodscus2)

This project is a sophisticated and efficient movie recommendation system with a **Streamlit front-end** that caters equally well to new users (Cold Start) and experienced users with established viewing histories.

## Features

- 🎯 **Content-Based Filtering**: For new users (Cold Start problem)
- 👥 **Collaborative Filtering**: For users with rating history
- 🔄 **Hybrid Approach**: Best of both worlds
- 🎨 **Beautiful UI**: Built with Streamlit for an intuitive user experience
- ⭐ **Movie Ratings**: Rate movies to get personalized recommendations
- 🔍 **Movie Search**: Browse and search through the movie collection
- 📊 **Rating History**: Track your rated movies and preferences

## Technologies Used

- **Python 3.x**
- **Streamlit**: Interactive web framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning (TF-IDF, cosine similarity)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AstronDaniel/Movie-Recommendation-Ai-Model.git
cd Movie-Recommendation-Ai-Model
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How It Works

### Content-Based Filtering
Uses TF-IDF vectorization and cosine similarity to find movies similar to ones you like based on:
- Genre
- Director
- Movie description

Perfect for new users who haven't rated many movies yet!

### Collaborative Filtering
Analyzes your rating history to recommend movies similar to those you rated highly. The more movies you rate, the better the recommendations become!

### Movie Rating System
Rate movies on a scale of 0.5 to 5.0 stars. Your ratings are used to:
- Build your user profile
- Generate personalized recommendations
- Improve recommendation accuracy over time

## Project Structure

```
Movie-Recommendation-Ai-Model/
├── app.py                 # Streamlit web application
├── movie_recommender.py   # Core recommendation engine
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## Screenshots

### Browse Movies
![Browse Movies](https://github.com/user-attachments/assets/98c6cf7e-9233-46a8-9c92-ee582236e60d)

### Get Recommendations
![Get Recommendations](https://github.com/user-attachments/assets/52b2315d-661f-415a-9c71-caf7723a030a)

### Rate Movies
![Rate Movies](https://github.com/user-attachments/assets/50a5ed70-1601-4776-864c-4423f3a01256)

## Future Enhancements

- Integration with real movie APIs (TMDB, OMDb)
- Larger movie dataset
- User authentication and persistent data storage
- Advanced recommendation algorithms (Matrix Factorization, Deep Learning)
- Movie posters and trailers
- Social features (share recommendations, see what friends are watching)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Movie data inspired by top-rated films from various sources
- Built with ❤️ using Streamlit and scikit-learn
