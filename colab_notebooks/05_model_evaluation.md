# 📈 Step 5: Model Evaluation & Export

## 🎯 **Objective**
Thoroughly evaluate our trained models, visualize performance, and export everything for web integration.

---

## 📋 **What We'll Do**
1. Evaluate on the held-out test set (final honest evaluation)
2. Calculate multiple metrics (RMSE, MAE, Precision, Recall)
3. Analyze diversity of recommendations
4. Compare model performance
5. Export final models for deployment

---

## 🤔 **Why This Evaluation Matters**

| What We Check | Why It's Important |
|---------------|-------------------|
| Test Set Performance | Honest estimate of real-world accuracy |
| Precision@K | Are recommended movies actually relevant? |
| Diversity | Are we avoiding repetitive suggestions? |
| Cold Start | Does model work for new users? |

---

## 💻 **Code Cells**

### **Cell 1: Load Everything**

```python
# ============================================
# LOAD TRAINED MODELS AND DATA
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from surprise import accuracy
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

# Paths (using Colab temp storage)
PROJECT_DIR = '/content/MovieRecommender'
DATA_DIR = f'{PROJECT_DIR}/data'
MODEL_DIR = f'{PROJECT_DIR}/models'
PLOTS_DIR = f'{PROJECT_DIR}/plots'

# Load data
train_df = pd.read_csv(f'{DATA_DIR}/train_ratings.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_ratings.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test_ratings.csv')
movies_df = pd.read_csv(f'{DATA_DIR}/movies_processed.csv')

# Load models
with open(f'{MODEL_DIR}/cf_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)

with open(f'{MODEL_DIR}/similarity_matrix.pkl', 'rb') as f:
    similarity_df = pickle.load(f)

with open(f'{MODEL_DIR}/hybrid_config.pkl', 'rb') as f:
    hybrid_config = pickle.load(f)

with open(f'{MODEL_DIR}/preprocessing_data.pkl', 'rb') as f:
    prep_data = pickle.load(f)

print("✅ All models and data loaded!")
print(f"   Test set size: {len(test_df):,} ratings")
```

---

### **Cell 2: Final Test Set Evaluation**

```python
# ============================================
# FINAL TEST SET EVALUATION
# ============================================
# This is the HONEST evaluation - no peeking at this data during training!

print("📊 FINAL TEST SET EVALUATION")
print("=" * 50)
print("⚠️ This is the definitive performance measure!")
print("=" * 50)

# Create test set in Surprise format
testset = [(row['user_id'], row['movie_id'], row['rating']) 
           for _, row in test_df.iterrows()]

# Get predictions
test_predictions = cf_model.test(testset)

# Calculate metrics
test_rmse = accuracy.rmse(test_predictions, verbose=False)
test_mae = accuracy.mae(test_predictions, verbose=False)

# Calculate additional stats
actuals = np.array([pred.r_ui for pred in test_predictions])
predictions = np.array([pred.est for pred in test_predictions])
errors = actuals - predictions

print(f"""
═══════════════════════════════════════════════════════
🏆 FINAL TEST SET RESULTS
═══════════════════════════════════════════════════════

PRIMARY METRICS
─────────────────────────────────────
RMSE:           {test_rmse:.4f}  {'✅' if test_rmse < 0.90 else '⚠️'}
MAE:            {test_mae:.4f}   {'✅' if test_mae < 0.75 else '⚠️'}
Target RMSE:    < 0.90

ERROR STATISTICS
─────────────────────────────────────
Mean Error:     {errors.mean():.4f} (should be ~0 for unbiased)
Std Error:      {errors.std():.4f}
Min Error:      {errors.min():.4f}
Max Error:      {errors.max():.4f}

INTERPRETATION
─────────────────────────────────────
• RMSE of {test_rmse:.4f} means predictions are off by ~{test_rmse:.2f} stars
• Mean error of {errors.mean():.4f} indicates {'unbiased' if abs(errors.mean()) < 0.05 else 'slightly biased'} predictions
• {'🎉 EXCELLENT!' if test_rmse < 0.85 else '✅ GOOD!' if test_rmse < 0.90 else '⚠️ Needs improvement'}

═══════════════════════════════════════════════════════
""")
```

---

### **Cell 3: Precision and Recall at K**

```python
# ============================================
# PRECISION@K AND RECALL@K
# ============================================
# These metrics focus on the TOP recommendations

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Calculate Precision@K and Recall@K for all users.
    
    Precision@K: Of the top K recommended, how many are relevant?
    Recall@K: Of all relevant items, how many are in top K?
    
    Relevant = actual rating >= threshold
    """
    user_est_true = defaultdict(list)
    
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))
    
    precisions = []
    recalls = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by predicted rating (descending)
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Number of relevant items
        n_relevant = sum(true >= threshold for (_, true) in user_ratings)
        
        if n_relevant == 0:
            continue
        
        # Get top K
        top_k = user_ratings[:k]
        
        # Count relevant in top K
        n_relevant_in_k = sum(true >= threshold for (_, true) in top_k)
        
        # Precision@K = relevant in top K / K
        precisions.append(n_relevant_in_k / k)
        
        # Recall@K = relevant in top K / total relevant
        recalls.append(n_relevant_in_k / n_relevant)
    
    return np.mean(precisions), np.mean(recalls)

print("📊 PRECISION AND RECALL ANALYSIS")
print("=" * 50)

# Calculate for different values of K
k_values = [5, 10, 15, 20]
results = []

for k in k_values:
    prec, rec = precision_recall_at_k(test_predictions, k=k, threshold=4.0)
    results.append({'K': k, 'Precision': prec, 'Recall': rec})
    print(f"K={k:2d}: Precision={prec:.4f}, Recall={rec:.4f}")

results_df = pd.DataFrame(results)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Precision plot
ax1 = axes[0]
ax1.bar(results_df['K'].astype(str), results_df['Precision'], color='steelblue')
ax1.set_xlabel('K (Number of Recommendations)', fontsize=12)
ax1.set_ylabel('Precision@K', fontsize=12)
ax1.set_title('Precision at Different K Values', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
for i, v in enumerate(results_df['Precision']):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')

# Recall plot
ax2 = axes[1]
ax2.bar(results_df['K'].astype(str), results_df['Recall'], color='coral')
ax2.set_xlabel('K (Number of Recommendations)', fontsize=12)
ax2.set_ylabel('Recall@K', fontsize=12)
ax2.set_title('Recall at Different K Values', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
for i, v in enumerate(results_df['Recall']):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/12_precision_recall.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"""
💡 INTERPRETATION
─────────────────────────────────────
• Precision@10 = {results_df[results_df['K']==10]['Precision'].values[0]:.2%} of our top 10 are relevant
• Recall@10 = {results_df[results_df['K']==10]['Recall'].values[0]:.2%} of all relevant items found in top 10
• Higher K → Higher recall but lower precision (tradeoff)
""")
```

**📝 Explanation:**
- **Precision@K**: "Are our recommendations actually good?"
- **Recall@K**: "Did we find most of the good movies?"
- There's always a tradeoff - more recommendations means lower precision

---

### **Cell 4: Diversity Analysis**

```python
# ============================================
# DIVERSITY ANALYSIS
# ============================================
# Check if recommendations are diverse enough

print("🎭 DIVERSITY ANALYSIS")
print("=" * 50)

# Recreate hybrid model
class HybridRecommender:
    def __init__(self, cf_model, similarity_matrix, movies_df, cf_weight=0.6):
        self.cf_model = cf_model
        self.similarity_matrix = similarity_matrix
        self.movies_df = movies_df
        self.cf_weight = cf_weight
        self.cb_weight = 1 - cf_weight
        
    def get_cf_predictions(self, user_id, movie_ids):
        predictions = {}
        for movie_id in movie_ids:
            try:
                pred = self.cf_model.predict(user_id, movie_id)
                predictions[movie_id] = pred.est
            except:
                predictions[movie_id] = 3.0
        return predictions
    
    def get_cb_scores(self, liked_movies, candidate_movies):
        if not liked_movies:
            return {m: 0.5 for m in candidate_movies}
        scores = {}
        for candidate in candidate_movies:
            if candidate in self.similarity_matrix.index:
                sims = []
                for liked in liked_movies:
                    if liked in self.similarity_matrix.columns:
                        sims.append(self.similarity_matrix.loc[candidate, liked])
                scores[candidate] = np.mean(sims) if sims else 0.5
            else:
                scores[candidate] = 0.5
        return scores
    
    def recommend(self, user_id, user_history=None, n=10):
        all_movies = set(self.movies_df['movie_id'].values)
        if user_history:
            watched = set([m[0] for m in user_history])
            candidate_movies = list(all_movies - watched)
        else:
            candidate_movies = list(all_movies)
        
        liked_movies = [m[0] for m in user_history if m[1] >= 4] if user_history else []
        cf_preds = self.get_cf_predictions(user_id, candidate_movies)
        cb_scores = self.get_cb_scores(liked_movies, candidate_movies)
        
        cf_values = list(cf_preds.values())
        cf_min, cf_max = min(cf_values), max(cf_values)
        if cf_max > cf_min:
            cf_norm = {k: (v - cf_min) / (cf_max - cf_min) for k, v in cf_preds.items()}
        else:
            cf_norm = {k: 0.5 for k in cf_preds}
        
        hybrid_scores = {}
        for movie_id in candidate_movies:
            hybrid_scores[movie_id] = (
                self.cf_weight * cf_norm.get(movie_id, 0.5) + 
                self.cb_weight * cb_scores.get(movie_id, 0.5)
            )
        
        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [m[0] for m in sorted_movies]

hybrid_model = HybridRecommender(cf_model, similarity_df, movies_df, cf_weight=0.6)

# Analyze diversity for sample users
def analyze_diversity(recommendations, similarity_matrix):
    """Calculate intra-list similarity (lower = more diverse)"""
    if len(recommendations) < 2:
        return 0
    
    sims = []
    for i in range(len(recommendations)):
        for j in range(i+1, len(recommendations)):
            m1, m2 = recommendations[i], recommendations[j]
            if m1 in similarity_matrix.index and m2 in similarity_matrix.columns:
                sims.append(similarity_matrix.loc[m1, m2])
    
    return np.mean(sims) if sims else 0

def count_genres(recommendations, movies_df):
    """Count unique genres in recommendations"""
    genres = set()
    for movie_id in recommendations:
        movie = movies_df[movies_df['movie_id'] == movie_id]
        if len(movie) > 0:
            movie_genres = movie.iloc[0]['genres'].split()
            genres.update(movie_genres)
    return len(genres)

# Analyze for multiple users
sample_users = train_df['user_id'].unique()[:50]
diversity_scores = []
genre_counts = []

for user_id in sample_users:
    user_ratings = train_df[train_df['user_id'] == user_id][['movie_id', 'rating']]
    user_history = list(user_ratings.itertuples(index=False, name=None))
    
    recs = hybrid_model.recommend(user_id, user_history, n=10)
    
    ils = analyze_diversity(recs, similarity_df)
    gc = count_genres(recs, movies_df)
    
    diversity_scores.append(ils)
    genre_counts.append(gc)

avg_ils = np.mean(diversity_scores)
avg_genres = np.mean(genre_counts)

print(f"""
═══════════════════════════════════════════════════════
📊 DIVERSITY METRICS
═══════════════════════════════════════════════════════

INTRA-LIST SIMILARITY (ILS)
─────────────────────────────────────
Average ILS:    {avg_ils:.4f}
Target:         < 0.70 (lower = more diverse)
Status:         {'✅ Good diversity!' if avg_ils < 0.70 else '⚠️ Too similar'}

GENRE DIVERSITY
─────────────────────────────────────
Avg Genres per 10 Recs: {avg_genres:.1f}
Target:                 > 4 genres
Status:                 {'✅ Good variety!' if avg_genres > 4 else '⚠️ Limited variety'}

═══════════════════════════════════════════════════════
""")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.hist(diversity_scores, bins=20, color='mediumpurple', edgecolor='white')
ax1.axvline(avg_ils, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_ils:.3f}')
ax1.axvline(0.70, color='green', linestyle='--', linewidth=2, label='Target: 0.70')
ax1.set_xlabel('Intra-List Similarity', fontsize=12)
ax1.set_ylabel('Number of Users', fontsize=12)
ax1.set_title('Diversity Distribution (Lower = Better)', fontsize=14, fontweight='bold')
ax1.legend()

ax2 = axes[1]
ax2.hist(genre_counts, bins=range(1, 15), color='teal', edgecolor='white')
ax2.axvline(avg_genres, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_genres:.1f}')
ax2.set_xlabel('Number of Genres in Recommendations', fontsize=12)
ax2.set_ylabel('Number of Users', fontsize=12)
ax2.set_title('Genre Variety per User', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/13_diversity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### **Cell 5: Cold Start Analysis**

```python
# ============================================
# COLD START ANALYSIS
# ============================================
# How well does our model handle new users?

print("❄️ COLD START ANALYSIS")
print("=" * 50)

# Find users in test set who have few ratings in training set
user_rating_counts = train_df.groupby('user_id').size()

# Categorize users
cold_users = user_rating_counts[user_rating_counts < 10].index.tolist()
warm_users = user_rating_counts[(user_rating_counts >= 10) & (user_rating_counts < 50)].index.tolist()
hot_users = user_rating_counts[user_rating_counts >= 50].index.tolist()

print(f"User Categories (by training ratings):")
print(f"  Cold (<10 ratings):   {len(cold_users)} users")
print(f"  Warm (10-50 ratings): {len(warm_users)} users")
print(f"  Hot (50+ ratings):    {len(hot_users)} users")

# Calculate RMSE for each category
def rmse_for_users(predictions, user_list):
    filtered = [p for p in predictions if p.uid in user_list]
    if not filtered:
        return np.nan
    squared_errors = [(p.r_ui - p.est) ** 2 for p in filtered]
    return np.sqrt(np.mean(squared_errors))

cold_rmse = rmse_for_users(test_predictions, cold_users)
warm_rmse = rmse_for_users(test_predictions, warm_users)
hot_rmse = rmse_for_users(test_predictions, hot_users)

print(f"""
═══════════════════════════════════════════════════════
📊 RMSE BY USER TYPE
═══════════════════════════════════════════════════════

Category        Ratings     RMSE
─────────────────────────────────────
Cold Users      < 10        {cold_rmse:.4f}
Warm Users      10-50       {warm_rmse:.4f}
Hot Users       50+         {hot_rmse:.4f}

💡 INSIGHTS
─────────────────────────────────────
• {'Cold users have higher error (expected)' if cold_rmse > hot_rmse else 'Model handles cold users well!'}
• Hot users benefit from more training data
• Our hybrid approach helps with cold start via content-based
═══════════════════════════════════════════════════════
""")

# Visualize
fig, ax = plt.subplots(figsize=(8, 5))

categories = ['Cold\n(<10 ratings)', 'Warm\n(10-50 ratings)', 'Hot\n(50+ ratings)']
rmse_values = [cold_rmse, warm_rmse, hot_rmse]
colors = ['#ff6b6b', '#feca57', '#48dbfb']

bars = ax.bar(categories, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Model Performance by User Activity Level', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(rmse_values) * 1.2)

for bar, rmse in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{rmse:.3f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/14_cold_start_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### **Cell 6: Comprehensive Model Comparison**

```python
# ============================================
# MODEL COMPARISON SUMMARY
# ============================================

print("📊 COMPREHENSIVE MODEL EVALUATION SUMMARY")
print("=" * 60)

# Collect all metrics
prec_10, rec_10 = precision_recall_at_k(test_predictions, k=10, threshold=4.0)

summary_data = {
    'Metric': [
        'Test RMSE',
        'Test MAE',
        'Precision@10',
        'Recall@10',
        'Avg Intra-List Similarity',
        'Avg Genre Diversity',
        'Cold User RMSE',
        'Hot User RMSE'
    ],
    'Value': [
        f'{test_rmse:.4f}',
        f'{test_mae:.4f}',
        f'{prec_10:.4f}',
        f'{rec_10:.4f}',
        f'{avg_ils:.4f}',
        f'{avg_genres:.1f}',
        f'{cold_rmse:.4f}',
        f'{hot_rmse:.4f}'
    ],
    'Target': [
        '< 0.90',
        '< 0.75',
        '> 0.40',
        '> 0.30',
        '< 0.70',
        '> 4',
        '< 1.10',
        '< 0.85'
    ],
    'Status': [
        '✅' if test_rmse < 0.90 else '❌',
        '✅' if test_mae < 0.75 else '❌',
        '✅' if prec_10 > 0.40 else '⚠️',
        '✅' if rec_10 > 0.30 else '⚠️',
        '✅' if avg_ils < 0.70 else '⚠️',
        '✅' if avg_genres > 4 else '⚠️',
        '✅' if cold_rmse < 1.10 else '⚠️',
        '✅' if hot_rmse < 0.85 else '⚠️'
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Visual summary
fig, ax = plt.subplots(figsize=(10, 6))

# Create a table visualization
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#4285f4'] * 4
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color status column
for i in range(len(summary_df) + 1):
    for j in range(4):
        cell = table[(i, j)]
        if i == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif j == 3:  # Status column
            if summary_df.iloc[i-1]['Status'] == '✅':
                cell.set_facecolor('#c8e6c9')
            elif summary_df.iloc[i-1]['Status'] == '⚠️':
                cell.set_facecolor('#fff3e0')
            else:
                cell.set_facecolor('#ffcdd2')

plt.title('Model Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/15_evaluation_summary.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### **Cell 7: Generate Example Recommendations**

```python
# ============================================
# EXAMPLE RECOMMENDATIONS SHOWCASE
# ============================================

print("🎬 EXAMPLE RECOMMENDATIONS SHOWCASE")
print("=" * 60)

# Pick 3 diverse users
example_users = [1, 100, 500]

for user_id in example_users:
    if user_id not in train_df['user_id'].values:
        continue
        
    user_ratings = train_df[train_df['user_id'] == user_id][['movie_id', 'rating']]
    user_history = list(user_ratings.itertuples(index=False, name=None))
    
    # Top rated movies by this user
    top_rated = user_ratings.nlargest(5, 'rating')
    
    print(f"\n{'='*60}")
    print(f"👤 USER {user_id}")
    print(f"{'='*60}")
    print(f"\n📺 Movies They Loved:")
    
    for _, row in top_rated.iterrows():
        movie = movies_df[movies_df['movie_id'] == row['movie_id']]
        if len(movie) > 0:
            print(f"   ⭐ {row['rating']:.0f} - {movie.iloc[0]['title'][:50]}")
    
    # Get recommendations
    recs = hybrid_model.recommend(user_id, user_history, n=5)
    
    print(f"\n🎯 Our Recommendations:")
    for i, movie_id in enumerate(recs, 1):
        movie = movies_df[movies_df['movie_id'] == movie_id]
        if len(movie) > 0:
            print(f"   {i}. {movie.iloc[0]['title'][:50]}")
            print(f"      Genres: {movie.iloc[0]['genres']}")
```

---

### **Cell 8: Export Final Models**

```python
# ============================================
# EXPORT FINAL MODELS FOR WEB DEPLOYMENT
# ============================================
# Using Colab's temp storage and files.download() for easy export

print("📦 EXPORTING MODELS FOR WEB DEPLOYMENT")
print("=" * 60)

import os
import shutil
from datetime import datetime
from google.colab import files  # For downloading files from Colab

# Create export directory in Colab temp storage
EXPORT_DIR = '/content/export'
os.makedirs(EXPORT_DIR, exist_ok=True)

# Export files
export_files = {
    'cf_model.pkl': cf_model,
    'similarity_matrix.pkl': similarity_df,
    'movies_data.pkl': movies_df[['movie_id', 'title', 'genres', 'year', 'avg_rating']],
    'preprocessing_data.pkl': prep_data,
    'hybrid_config.pkl': hybrid_config
}

for filename, data in export_files.items():
    filepath = f'{EXPORT_DIR}/{filename}'
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    size = os.path.getsize(filepath) / 1024 / 1024  # MB
    print(f"✅ {filename}: {size:.2f} MB")

# Create metadata
metadata = {
    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'precision_at_10': prec_10,
    'recall_at_10': rec_10,
    'diversity_score': avg_ils,
    'n_users': train_df['user_id'].nunique(),
    'n_movies': len(movies_df),
    'n_ratings': len(train_df),
    'model_type': 'hybrid_svd_content',
    'cf_weight': hybrid_config['cf_weight']
}

with open(f'{EXPORT_DIR}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n✅ metadata.pkl exported")

# Create README for export
export_readme = f"""
# Movie Recommender - Exported Models

## Export Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files
- `cf_model.pkl` - SVD collaborative filtering model
- `similarity_matrix.pkl` - Movie similarity matrix for content-based
- `movies_data.pkl` - Movie information (id, title, genres, etc.)
- `preprocessing_data.pkl` - TF-IDF vectorizer and mappings
- `hybrid_config.pkl` - Model configuration

## Performance Metrics
- Test RMSE: {test_rmse:.4f}
- Test MAE: {test_mae:.4f}
- Precision@10: {prec_10:.4f}
- Recall@10: {rec_10:.4f}

## Usage
```python
import pickle

# Load models
cf_model = pickle.load(open('cf_model.pkl', 'rb'))
similarity_df = pickle.load(open('similarity_matrix.pkl', 'rb'))
movies_df = pickle.load(open('movies_data.pkl', 'rb'))

# Make prediction
prediction = cf_model.predict(user_id=1, movie_id=50)
print(f"Predicted rating: {{prediction.est}}")
```
"""

with open(f'{EXPORT_DIR}/README.md', 'w') as f:
    f.write(export_readme)

print("\n✅ README.md created")

# Zip all export files for easy download
shutil.make_archive('/content/movie_recommender_models', 'zip', EXPORT_DIR)
print("\n📦 Created: movie_recommender_models.zip")

print(f"""
═══════════════════════════════════════════════════════
📦 EXPORT COMPLETE
═══════════════════════════════════════════════════════

Export Location: {EXPORT_DIR}

Files Ready for Download:
─────────────────────────────────────
📁 export/
   ├── cf_model.pkl
   ├── similarity_matrix.pkl
   ├── movies_data.pkl
   ├── preprocessing_data.pkl
   ├── hybrid_config.pkl
   ├── metadata.pkl
   └── README.md

🌐 NEXT STEPS:
1. Run the download cell below to get the ZIP file
2. Extract and place files in your web app's models/ directory
3. Load models with pickle
4. Integrate with Flask/Streamlit

═══════════════════════════════════════════════════════
""")

# ============================================
# DOWNLOAD MODELS (Run this to trigger download)
# ============================================
print("⬇️ Downloading models ZIP file...")
files.download('/content/movie_recommender_models.zip')
print("✅ Download started! Check your browser downloads.")
```

---

### **Cell 9: Final Summary Visualization**

```python
# ============================================
# FINAL PROJECT SUMMARY
# ============================================

print("=" * 60)
print("🎉 MOVIE RECOMMENDATION SYSTEM - TRAINING COMPLETE!")
print("=" * 60)

# Create final summary visualization
fig = plt.figure(figsize=(14, 10))

# Subplot 1: Performance radar chart (simplified)
ax1 = fig.add_subplot(2, 2, 1)
metrics = ['RMSE\n(inverted)', 'MAE\n(inverted)', 'Precision', 'Recall', 'Diversity']
values = [
    1 - test_rmse/1.5,  # Invert and normalize
    1 - test_mae/1.0,
    prec_10,
    rec_10,
    1 - avg_ils  # Invert (lower is better)
]
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

ax1.plot(angles, values, 'o-', linewidth=2, color='#4285f4')
ax1.fill(angles, values, alpha=0.25, color='#4285f4')
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metrics)
ax1.set_ylim(0, 1)
ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold')

# Subplot 2: Training pipeline
ax2 = fig.add_subplot(2, 2, 2)
steps = ['Data\nLoaded', 'EDA\nComplete', 'Preprocessing\nDone', 'Model\nTrained', 'Evaluation\nPassed']
ax2.barh(steps, [1, 1, 1, 1, 1], color=['#4CAF50']*5)
ax2.set_xlim(0, 1.2)
for i, s in enumerate(steps):
    ax2.text(1.05, i, '✅', fontsize=20, va='center')
ax2.set_title('Training Pipeline Status', fontsize=14, fontweight='bold')
ax2.set_xlabel('Completion')

# Subplot 3: Key statistics
ax3 = fig.add_subplot(2, 2, 3)
ax3.axis('off')
stats_text = f"""
📊 DATASET STATISTICS
──────────────────────
Users:      {train_df['user_id'].nunique():,}
Movies:     {len(movies_df):,}
Ratings:    {len(train_df) + len(val_df) + len(test_df):,}

🎯 MODEL PERFORMANCE
──────────────────────
Test RMSE:      {test_rmse:.4f}
Test MAE:       {test_mae:.4f}
Precision@10:   {prec_10:.4f}
Recall@10:      {rec_10:.4f}

🔧 MODEL CONFIGURATION
──────────────────────
CF Weight:      {hybrid_config['cf_weight']*100:.0f}%
CB Weight:      {(1-hybrid_config['cf_weight'])*100:.0f}%
SVD Factors:    {hybrid_config['best_params']['n_factors']}
"""
ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.set_title('Summary Statistics', fontsize=14, fontweight='bold')

# Subplot 4: Learning type explanation
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')
learning_text = """
🎓 LEARNING TYPE: SUPERVISED LEARNING
═══════════════════════════════════════════

WHY SUPERVISED?
• We have LABELED DATA (actual user ratings)
• Model learns by comparing predictions to real ratings
• Goal: Minimize prediction error (RMSE)

HOW IT WORKS:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ User-Movie  │───▶│    SVD     │───▶│  Predicted  │
│   Pairs     │    │   Model    │    │   Rating    │
└─────────────┘    └─────────────┘    └─────────────┘
                         ▲                  │
                         │    Compare       ▼
                   ┌─────┴─────┐    ┌─────────────┐
                   │  Update   │◀───│   Actual    │
                   │  Weights  │    │   Rating    │
                   └───────────┘    └─────────────┘
"""
ax4.text(0.05, 0.5, learning_text, fontsize=10, family='monospace', va='center')
ax4.set_title('Learning Approach', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/16_final_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
═══════════════════════════════════════════════════════════════
🎉 CONGRATULATIONS! TRAINING COMPLETE!
═══════════════════════════════════════════════════════════════

You have successfully:
✅ Explored and understood the MovieLens dataset
✅ Preprocessed data for machine learning
✅ Trained a collaborative filtering model (SVD)
✅ Built a content-based similarity model
✅ Combined both into a hybrid recommender
✅ Evaluated performance with multiple metrics
✅ Exported models for web deployment

NEXT STEPS:
1. Download the ZIP file (auto-triggered above)
2. Set up your Flask/Streamlit web app
3. Load the pickle files
4. Start recommending movies! 🎬

═══════════════════════════════════════════════════════════════
""")
```

---

## 📊 **Evaluation Metrics Summary**

```mermaid
graph LR
    subgraph Accuracy ["📏 Accuracy Metrics"]
        A[RMSE] --> B[Lower = Better]
        C[MAE] --> B
    end
    
    subgraph Ranking ["📊 Ranking Metrics"]
        D[Precision@K] --> E[Higher = Better]
        F[Recall@K] --> E
    end
    
    subgraph Diversity ["🎭 Diversity Metrics"]
        G[Intra-List Similarity] --> H[Lower = Better]
        I[Genre Coverage] --> J[Higher = Better]
    end
    
    style A fill:#e3f2fd
    style D fill:#f3e5f5
    style G fill:#e8f5e9
```

---

## 🎉 **Project Complete!**

You now have:
1. **Trained Models** - Ready for deployment
2. **Performance Metrics** - Documented and validated
3. **Export Files** - Downloaded as a ZIP file

### Download Your Models:
The ZIP file `movie_recommender_models.zip` was automatically downloaded when you ran Cell 8. Extract it to get all the model files!

---

*Notebook 5 of 5 | Movie Recommendation System Training - COMPLETE! 🎬*
