
# Movie Recommender - Exported Models

## Export Date
2025-12-04 10:13:20

## Files
- `cf_model.pkl` - SVD collaborative filtering model
- `similarity_matrix.pkl` - Movie similarity matrix for content-based
- `movies_data.pkl` - Movie information (id, title, genres, etc.)
- `preprocessing_data.pkl` - TF-IDF vectorizer and mappings
- `hybrid_config.pkl` - Model configuration

## Performance Metrics
- Test RMSE: 0.9157
- Test MAE: 0.7234
- Precision@10: 0.5405
- Recall@10: 0.8062

## Usage
```python
import pickle

# Load models {#load-models  data-source-line="718"}
cf_model = pickle.load(open('cf_model.pkl', 'rb'))
similarity_df = pickle.load(open('similarity_matrix.pkl', 'rb'))
movies_df = pickle.load(open('movies_data.pkl', 'rb'))

# Make prediction {#make-prediction  data-source-line="723"}
prediction = cf_model.predict(user_id=1, movie_id=50)
print(f"Predicted rating: {prediction.est}")
``` {data-source-line="726"}
