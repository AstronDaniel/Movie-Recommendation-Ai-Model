# 📦 Step 1: Setup and Installation

## 🎯 **Objective**
Set up our Google Colab environment with all the necessary libraries and tools to build our movie recommendation system.

---

## 📋 **What We'll Do in This Notebook**
1. Install required Python packages
2. Import all necessary libraries
3. Set up project folders (in Colab's temp storage)
4. Download the MovieLens dataset
5. Verify everything works correctly

---

## 🤔 **Why These Specific Libraries?**

| Library | Purpose | Why We Need It |
|---------|---------|----------------|
| `pandas` | Data manipulation | Load and process our movie/rating datasets |
| `numpy` | Numerical operations | Fast mathematical computations |
| `matplotlib` & `seaborn` | Visualization | Create plots to understand our data |
| `scikit-learn` | ML utilities | Train-test split, metrics, TF-IDF |
| `surprise` | Recommendation algorithms | SVD for collaborative filtering |
| `scipy` | Scientific computing | Sparse matrices, cosine similarity |

---

## 💻 **Code Cells**

### **Cell 1: Install Required Packages**

```python
# ============================================
# INSTALL REQUIRED PACKAGES
# ============================================
# These packages are not pre-installed in Colab
# We use !pip to install them directly

# Surprise library - specialized for recommendation systems
# Contains SVD, KNN, and other collaborative filtering algorithms
!pip install scikit-surprise

# For progress bars during long operations
!pip install tqdm

print("✅ All packages installed successfully!")
```

**📝 Explanation:**
- `scikit-surprise` is a Python library specifically designed for building recommendation systems
- It provides ready-to-use algorithms like SVD (Singular Value Decomposition)
- `tqdm` gives us nice progress bars so we know how long operations take

---

### **Cell 2: Import All Libraries**

```python
# ============================================
# IMPORT LIBRARIES
# ============================================

# --- Data Manipulation ---
import pandas as pd                    # DataFrames for tabular data
import numpy as np                     # Numerical operations

# --- Visualization ---
import matplotlib.pyplot as plt        # Basic plotting
import seaborn as sns                  # Beautiful statistical plots
from matplotlib.ticker import FuncFormatter  # Format axis labels

# --- Machine Learning (Scikit-Learn) ---
from sklearn.model_selection import train_test_split  # Split data
from sklearn.feature_extraction.text import TfidfVectorizer  # Text to vectors
from sklearn.metrics.pairwise import cosine_similarity  # Similarity calculation
from sklearn.preprocessing import MinMaxScaler  # Normalize values

# --- Recommendation System (Surprise) ---
from surprise import Dataset, Reader  # Load data for Surprise
from surprise import SVD, SVDpp, NMF  # Collaborative filtering algorithms
from surprise import accuracy         # Evaluation metrics
from surprise.model_selection import cross_validate, train_test_split as surprise_split

# --- Utilities ---
import warnings
warnings.filterwarnings('ignore')      # Hide unnecessary warnings
from tqdm import tqdm                  # Progress bars
import pickle                          # Save/load models
import os                              # File operations

# --- Set Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print(f"📊 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")
```

**📝 Explanation:**
- We organize imports by category (data, visualization, ML, etc.)
- `warnings.filterwarnings('ignore')` hides warning messages that can clutter our output
- Setting a plot style makes all our visualizations consistent and professional

---

### **Cell 3: Set Up Project Folders**

```python
# ============================================
# SET UP PROJECT FOLDERS
# ============================================
# Create folder structure in Colab's temporary storage
# Note: Files will be deleted when session ends
# Make sure to download important models before closing!

import os

# Project directories (in Colab's temp storage)
PROJECT_DIR = '/content/MovieRecommender'
DATA_DIR = f'{PROJECT_DIR}/data'
MODEL_DIR = f'{PROJECT_DIR}/models'
PLOTS_DIR = f'{PROJECT_DIR}/plots'

# Create directories if they don't exist
for directory in [PROJECT_DIR, DATA_DIR, MODEL_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"📁 Created: {directory}")

print("\n✅ Project folders ready!")
print("⚠️ Remember: Download your models before the session ends!")
```

**📝 Explanation:**
- We create folders in Colab's `/content/` directory (temporary storage)
- Folder structure:
  - `data/` - Store downloaded datasets
  - `models/` - Save trained models (.pkl files)
  - `plots/` - Save visualizations
- **Important**: Colab sessions reset after ~12 hours of inactivity - download your trained models before closing!

---

### **Cell 4: Download MovieLens Dataset**

```python
# ============================================
# DOWNLOAD MOVIELENS DATASET
# ============================================
# We're using MovieLens 100K for faster experimentation
# You can switch to MovieLens 25M for production

import urllib.request
import zipfile

# MovieLens 100K - Good for learning and experimentation
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = f"{DATA_DIR}/ml-100k.zip"
EXTRACT_PATH = f"{DATA_DIR}/ml-100k"

# Download if not already downloaded
if not os.path.exists(EXTRACT_PATH):
    print("📥 Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
    
    print("📦 Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Clean up zip file
    os.remove(ZIP_PATH)
    print("✅ Dataset downloaded and extracted!")
else:
    print("✅ Dataset already exists, skipping download.")

# List contents
print("\n📂 Dataset contents:")
for file in os.listdir(EXTRACT_PATH):
    file_path = os.path.join(EXTRACT_PATH, file)
    size = os.path.getsize(file_path) / 1024  # KB
    print(f"   {file}: {size:.1f} KB")
```

**📝 Explanation:**
- **MovieLens 100K**: Contains 100,000 ratings from 943 users on 1,682 movies
- Perfect size for learning - fast to train but still realistic
- Later, you can upgrade to MovieLens 25M (25 million ratings) for production

---

### **Cell 5: Verify Setup - Quick Data Preview**

```python
# ============================================
# VERIFY SETUP - QUICK DATA PREVIEW
# ============================================

# Load ratings data
ratings_path = f"{EXTRACT_PATH}/u.data"
ratings_df = pd.read_csv(
    ratings_path, 
    sep='\t', 
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    encoding='latin-1'
)

# Load movie data
movies_path = f"{EXTRACT_PATH}/u.item"
movies_df = pd.read_csv(
    movies_path,
    sep='|',
    encoding='latin-1',
    names=['movie_id', 'title', 'release_date', 'video_release', 'imdb_url',
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

# Quick summary
print("=" * 50)
print("📊 SETUP VERIFICATION - DATA LOADED SUCCESSFULLY")
print("=" * 50)
print(f"\n🎬 Total Movies: {len(movies_df):,}")
print(f"👥 Total Users: {ratings_df['user_id'].nunique():,}")
print(f"⭐ Total Ratings: {len(ratings_df):,}")
print(f"📈 Ratings Range: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
print(f"📊 Average Rating: {ratings_df['rating'].mean():.2f}")

print("\n🔍 Sample Ratings:")
print(ratings_df.head())

print("\n🎥 Sample Movies:")
print(movies_df[['movie_id', 'title', 'Action', 'Comedy', 'Drama']].head())

print("\n" + "=" * 50)
print("✅ SETUP COMPLETE - READY FOR DATA EXPLORATION!")
print("=" * 50)
```

**📝 Explanation:**
- This cell verifies everything is working correctly
- We load both datasets and display key statistics
- If you see the summary without errors, the setup is successful!

---

## 📊 **Expected Output**

After running all cells, you should see:
- ✅ All packages installed
- ✅ All libraries imported
- ✅ Project folders created
- ✅ Dataset downloaded (or already exists)
- ✅ Data preview showing ~100K ratings

---

## ⚠️ **Troubleshooting**

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Re-run the installation cell |
| Download fails | Check your internet connection |
| Memory error | Restart runtime (Runtime → Restart) |

---

## 💾 **Saving Your Work**

Since we're using Colab's temporary storage:
1. **Download models** at the end using: `files.download('model.pkl')`
2. Or **save to Drive** manually if needed
3. Re-run notebooks from the start if session resets

---

## ➡️ **Next Step**
Proceed to **02_data_exploration.md** to explore and understand our dataset in depth!

---

*Notebook 1 of 5 | Movie Recommendation System Training*
