# Movie API Setup Guide

This application uses **TMDB (The Movie Database) API** to fetch movie posters and artwork.

## Getting Your API Key

1. Go to [TMDB Website](https://www.themoviedb.org/)
2. Create a free account or sign in
3. Navigate to [API Settings](https://www.themoviedb.org/settings/api)
4. Request an API key (it's free!)
5. Copy your API key

## Setting Up the API Key

### Option 1: Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
$env:TMDB_API_KEY="your_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set TMDB_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export TMDB_API_KEY="your_api_key_here"
```

### Option 2: Modify movie_api.py

You can also directly set the API key in `movie_api.py`:

```python
def __init__(self, api_key: Optional[str] = None):
    self.api_key = api_key or os.getenv('TMDB_API_KEY', 'your_api_key_here')
```

## Features

- **Movie Posters**: Automatically fetches high-quality movie posters
- **Caching**: Poster URLs are cached to reduce API calls
- **Fallback**: If a poster isn't found, displays a placeholder with the movie's first letter

## API Limits

TMDB API is free and has generous rate limits:
- 40 requests per 10 seconds
- The app uses caching to minimize API calls

## Troubleshooting

If posters don't appear:
1. Check that your API key is set correctly
2. Verify your internet connection
3. Check the console for error messages
4. The app will fall back to letter placeholders if API is unavailable

