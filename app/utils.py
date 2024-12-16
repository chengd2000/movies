import requests
import os

API_KEY = os.getenv('API_KEY', 'e9d2e8f44b5d1e54fa22022b4a771f6c')
BASE_URL = "https://api.themoviedb.org/3"

def fetch_movies(category="popular", search_query=None):
    if search_query:
        endpoint = f"{BASE_URL}/search/movie"
        params = {"api_key": API_KEY, "query": search_query}
    else:
        endpoint = f"{BASE_URL}/movie/{category}"
        params = {"api_key": API_KEY}

    response = requests.get(endpoint, params=params)
    return response.json().get('results', [])

def fetch_movie_details(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY, "append_to_response": "credits"}
    response = requests.get(endpoint, params=params)
    return response.json()

