from flask import Blueprint, render_template, request, jsonify
import requests

main = Blueprint('main', __name__)

API_KEY = 'e9d2e8f44b5d1e54fa22022b4a771f6c'
BASE_URL = 'https://api.themoviedb.org/3'

def fetch_movies(search_query=None):
    endpoint = f"{BASE_URL}/search/movie" if search_query else f"{BASE_URL}/movie/popular"
    params = {'api_key': API_KEY, 'query': search_query} if search_query else {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data.get('results', [])

@main.route('/')
def home():
    movies = fetch_movies()
    return render_template('index.html', movies=movies)

@main.route('/search', methods=['GET'])
def search_movies_api():
    query = request.args.get('query', '')
    if query:
        movies = fetch_movies(search_query=query)
        return jsonify(movies)
    return jsonify([])

@main.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    movie = response.json()

    if 'credits' not in movie:
        movie['credits'] = {'cast': []}  # הגדרת ברירת מחדל אם אין קרדיטים

    return render_template('movie_details.html', movie=movie)

