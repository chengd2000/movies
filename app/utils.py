import requests # type: ignore
import os
from flask import Blueprint, json, render_template, request, jsonify
import pandas as pd
from app.ML import *
import json
from dotenv import load_dotenv

load_dotenv()  # טוען את המשתנים מהקובץ .env

API_KEY = os.getenv('API_KEY')

BASE_URL = "https://api.themoviedb.org/3"

def fetch_movies(search_query=None):
    endpoint = f"{BASE_URL}/search/movie" if search_query else f"{BASE_URL}/movie/popular"
    url_endpoint=f"{BASE_URL}/search/movie?query={search_query}&language=en-US&api_key={API_KEY}" if search_query else f"{BASE_URL}/movie/pupular?api_key={API_KEY}"
    params = {'api_key': API_KEY, 'query': search_query} if search_query else {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    data = response.json()
    returned_data=data.get('results', [])
    return returned_data, url_endpoint

def print_movies_list(df=None,genres=[],from_date=None,until_date=None,min_rating=None,actor1=None,actor2=None):
    # json_data_incoming = json.dumps(data)
    # json_data_incoming =data
    # df = pd.read_json(json_data_incoming)
    # df= pd.DataFrame(data)
    #filtered_df=transform_data(df)
    if(not df.empty):
        
        if(genres):
            for genre in genres:
                    if(genre in df.columns):
                        df = df[df[genre]==1]
        if(from_date):
            df = df[df["release_date"]>=from_date]
        if(until_date):
            df = df[df["release_date"]<=until_date]
        if(min_rating):
            df = df[df["vote_average"]>= float(min_rating)]
        if(actor1 or actor2):
            true_rows_ids=[]
            for row in df.itertuples(index=True, name='Pandas'):  
                actor1_found=False
                actor2_found=False              
                all_actors_in_movie=[]
                actors_data = fetch_movie_credits(row.id)['cast']
                # #a_df = pd.DataFrame(actors_data)
                # all_actors_in_movie=actors_data.cast
                #print(actors_data)
                for actor_in_movie in actors_data:
                    if actor1==actor_in_movie['name'] and actor1!="":
                        actor1_found=True
                    if actor2==actor_in_movie['name'] and actor2!="":
                        actor2_found=True
                
                if actor1_found==True and actor2_found==False:
                    print(f"{actor1} is in {row.title}")
                    if actor2=="":
                        true_rows_ids.append(row.id)
                if actor2_found==True and actor1_found==False:
                    print(f"{actor2} is in {row.title}")
                    if actor1=="":
                        true_rows_ids.append(row.id)
                if actor1_found==True and actor2_found==True:
                    print(f"Both {actor1} and {actor2} are in {row.title}")
                    if actor1!="" and actor2!="":
                        true_rows_ids.append(row.id)
            print(true_rows_ids)
            df = df[df['id'].isin(true_rows_ids)]
    if df.empty: 
        print("\nNo rows match the filter condition.")
        json_data_out = json.dumps({"results": []}) 
    else: 
        json_data_out = df.to_json(orient="records")
    json_object = json.loads(json_data_out)
    result = json_object
    return result

def fetch_popular_movies():
    endpoint = f"{BASE_URL}/movie/popular?page=1"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data.get('results', [])

def fetch_trending_movies(time='week'):
    endpoint = f"{BASE_URL}/trending/movie/{time}"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data.get('results', [])

def fetch_upcoming_movies(country='IL'):
    endpoint = f"{BASE_URL}/movie/upcoming?region={country}"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data.get('results', [])

def fetch_top_rated_movies():
    endpoint = f"{BASE_URL}/movie/top_rated"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data.get('results', [])


def fetch_movie_credits(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}/credits?language=en-US"
    params = {'api_key': API_KEY, 'query': movie_id} if movie_id else {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    return response.json()
    return data.get('results', [])

def fetch_reviews(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}/reviews?language=en-US&page=1"
    params = {'api_key': API_KEY, 'query': movie_id} if movie_id else {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    return response.json()
    return data.get('results', [])  

def fetch_movie_details(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    return response.json()
    return data.get('results', [])

def fetch_movie_trailer(movie_id): 
    endpoint = f"{BASE_URL}/movie/{movie_id}/videos"
    params = {'api_key': API_KEY} 
    response = requests.get(endpoint, params=params) 
    data = response.json() 
    for video in data.get('results', []): 
        if video['site'] == 'YouTube' and video['type'] == 'Trailer': 
            return f"https://www.youtube.com/embed/{video['key']}" 
    return None

def fetch_movie_images(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}/images?language=en"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    return response.json()
    return data.get('results', [])

def fetch_similar_movies(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}/recommendations"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    return response.json()
    return data.get('results', []) 

def fetch_genres():
    endpoint = f"{BASE_URL}/genre/movie/list"
    params = {'api_key': API_KEY}
    response = requests.get(endpoint, params=params)
    return response.json()
    return data.get('results', [])  

def fetch_top_actors(total_pages):
    #endpoint = f"{BASE_URL}/person/popular"
    #params = {'api_key': API_KEY}
    all_data = []
    # response = requests.get(f"{BASE_URL}/person/popular?api_key={API_KEY}")
    # print(response)
    for page in range(1, total_pages + 1):
        print(f"משיכת שחקנים מעמוד {page}...")
        response = requests.get(f"{BASE_URL}/person/popular?page={page}&api_key={API_KEY}")
        if response.status_code == 200:
            data = response.json().get("results", [])
            df = pd.DataFrame(data)
            all_data.append(df) 
        else:
            print(f"שגיאה בשליפת שחקנים מהעמוד {page}: {response.status_code}")
            break
    if response.status_code == 200:
        data = response.json().get("results", [])

        df = pd.DataFrame(data)
        all_data.append(df) 
    if(all_data):
        res=pd.concat(all_data, ignore_index=True)   
        json_data_out = res.to_json(orient="records")
    else: 
        print("\nNo rows match the filter condition.")
        json_data_out = json.dumps({"results": []}) 
    json_object = json.loads(json_data_out)
    result = json_object
    return result

    #return response.json()
    #return data.get('results', [])  