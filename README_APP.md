**Cineflow — Flask UI for Movie Recommender**

- Run the Flask app locally and it will try to load pickled models from the `models/` folder.
- Designed with a dark Netflix-like aesthetic (unique colors and layout).

Setup
1. Create a Python environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place your exported model files in the `models/` folder. Expected filenames (recommended):
- `cf_model.pkl`
- `similarity_matrix.pkl`
- `movies_data.pkl` (pandas DataFrame with columns: `movie_id`, `title`, `genres`, `year`, `avg_rating`)
- `preprocessing_data.pkl`
- `hybrid_config.pkl`

3. Run the app

```powershell
python app.py
```

4. Open `http://localhost:5000` in your browser. Enter a `user_id` and press "Get Recommendations".

Notes
- The app is defensive: if models are missing it will show placeholders and instructions.
- The collaborative model (if present) will be used to predict top movies by estimated rating.
- This UI is intentionally minimal so you can iterate quickly — let me know if you want additional pages (movie detail, authentication, posters).
