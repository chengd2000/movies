from flask import Blueprint, render_template, request, jsonify
from app.ML import *
from app.utils import *
main = Blueprint('main', __name__)
import datetime
from dotenv import load_dotenv

load_dotenv()  # טוען את המשתנים מהקובץ .env

API_KEY = os.getenv('API_KEY')

BASE_URL = 'https://api.themoviedb.org/3'

@main.route('/')
def home():
    popular_movies=fetch_popular_movies()
    trending_movies_week=fetch_trending_movies()
    trending_movies_day=fetch_trending_movies('day')
    upcoming_movies_israel=fetch_upcoming_movies()
    upcoming_movies_usa=fetch_upcoming_movies('US')
    upcoming_movies_japan=fetch_upcoming_movies('JP')
    upcoming_movies_spain=fetch_upcoming_movies('US')
    upcoming_movies_italy=fetch_upcoming_movies('IT')
    upcoming_movies_australia=fetch_upcoming_movies('AU')
    upcoming_movies_china=fetch_upcoming_movies('CN')
    upcoming_movies_russia=fetch_upcoming_movies('RU')
    upcoming_movies_india=fetch_upcoming_movies('IN')
    top_rated_movies=fetch_top_rated_movies()
    genres=fetch_genres()
    top_actors=fetch_top_actors(10)
    return render_template('index.html', popular_movies=popular_movies,trending_movies_week=trending_movies_week,trending_movies_day=trending_movies_day,upcoming_movies_israel=upcoming_movies_israel,upcoming_movies_usa=upcoming_movies_usa, upcoming_movies_japan=upcoming_movies_japan,upcoming_movies_spain=upcoming_movies_spain,upcoming_movies_italy=upcoming_movies_italy,upcoming_movies_australia=upcoming_movies_australia,upcoming_movies_china=upcoming_movies_china,upcoming_movies_russia=upcoming_movies_russia,upcoming_movies_india=upcoming_movies_india,top_rated_movies=top_rated_movies,genres=genres,top_actors=top_actors)

@main.route('/search')
def search_movies_api():
    query = request.args.get('query', '')
    genres = request.args.get('genres', '')
    from_date = request.args.get('from_date', '')
    until_date= request.args.get('until_date', '')
    min_rating=request.args.get('min_rating', '')
    actor1=request.args.get('actor1', '')
    actor2=request.args.get('actor2', '')
    url=f"{BASE_URL}/movie/popular?api_key={API_KEY}"
    if(genres):
        if "_" in genres:
            genres= genres.replace('_',' ')
        if "," in genres:
            genres = genres.split(",")
    if (query):
        searced_movies,url=fetch_movies(search_query=query)
        if searced_movies!={"results":[]}:
        #response = requests.get(url)
        #if response.status_code == 200:
            special_movie=print_movies_list(fetch_all_pages(url,30),genres=genres,from_date=from_date,until_date=until_date,min_rating=min_rating,actor1=actor1,actor2=actor2)
            #print(special_movie)
        # return jsonify(special_movie)
    else:
        
        special_movie=print_movies_list(fetch_all_pages(url,50),genres=genres,from_date=from_date,until_date=until_date,min_rating=min_rating,actor1=actor1,actor2=actor2)
        #print(special_movie)
    #print(response.status_code)
    #if special_movie!=None:
    #print("results:")
    #print(special_movie)
        #response = requests.get(special_movie)
        #if response.status_code == 200:
    if special_movie:
        if not special_movie=={"results":[]}:
            return render_template("search.html",movies=special_movie)
    return render_template("search.html")



@main.route('/Trending_Day')
def Trending_Day():
    trending_movies_day=fetch_trending_movies('day')
    return render_template("Trending_Day.html",trending_movies_day=trending_movies_day)

@main.route('/Trending_Week')
def Trending_Week():
    trending_movies_week=fetch_trending_movies()
    return render_template("Trending_Week.html",trending_movies_week=trending_movies_week)

@main.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    movie = fetch_movie_details(movie_id=movie_id)
    # #Check for consistent array lengths
    # lengths = [len(value) for value in movie.values()] 
    # if len(set(lengths)) != 1: raise ValueError("All arrays must be of the same length")
    # #Flatten the inner arrays and create a DataFrame 
    # flattened_data = {key: [] for key in movie} 
    # for key, values in movie.items(): 
    #     for value in values: 
    #         if isinstance(value, list): 
    #             for subvalue in value: flattened_data[key].append(subvalue) 
    #             else: flattened_data[key].append(value) 
    # # # Create the DataFrame'
    # print(flattened_data)
    # df = pd.DataFrame(flattened_data)
    
    #movie=transform_data(df)
    credits = fetch_movie_credits(movie_id=movie_id)
    trailer = fetch_movie_trailer(movie_id=movie_id)
    images = fetch_movie_images(movie_id=movie_id)
   # similar_movies = fetch_similar_movies(movie_id=movie_id)
    reviews=fetch_reviews(movie_id=movie_id)
    return render_template('movie_details.html', movie=movie, credits=credits, trailer=trailer,images=images,reviews=reviews)

    
# מסלול Flask
@main.route('/graphs')
def fetch_data_for_algorithm():
    print("enterd function")
    total_pages = 100  # מספר העמודים שברצונך למשוך
    url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"

    movies_data =fetch_all_pages(url,total_pages)   # שליפת נתונים

    # יצירת היסטוגרמה והחזרת נתוני האלגוריתם
    graph_json = diagram(movies_data)
    graph_json1 = diagram1(movies_data)
    graph_json2 = diagram2(movies_data)
    graph_json3 = diagram3(movies_data)
    graph_json4 = diagram4(movies_data)
    graph_json5 = diagram5(movies_data)
    graph_json6 = diagram6(movies_data)
    graph_json7 = diagram7(movies_data)
    graph_json8 = diagram8(movies_data)
    graph_json9 = diagram9(movies_data)
    graph_json10 = diagram10(movies_data)
    graph_json11 = diagram11(movies_data)
    graph_json12 = diagram12(movies_data)
    graph_json13 = diagram13(movies_data)
    graph_json14 = diagram14(movies_data)
    graph_json15 = diagram15(movies_data)
    # predictions = movies_data  
    #return jsonify({"histogram": histogram_json})
    return render_template('graphs.html',graph_json = graph_json, graph_json1 = graph_json1, graph_json2 = graph_json2, graph_json3 = graph_json3, graph_json4 = graph_json4, graph_json5 = graph_json5, graph_json6 = graph_json6, graph_json7 = graph_json7, graph_json8 = graph_json8, graph_json9 = graph_json9, graph_json10 = graph_json10, graph_json11 = graph_json11, graph_json12 = graph_json12, graph_json13 = graph_json13, graph_json14 = graph_json14, graph_json15 = graph_json15)
    

@main.route('/MLalgo')
def fetch_data_for_ML():
    total_pages = 100
    url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"
    movies_data = fetch_all_pages(url, total_pages)
    histogram_json = perform_linear_regression(movies_data)
    histogram_json2 = RandomForest(movies_data)
    histogram_json3 = k_means(movies_data)
    histogram_json4 = k_means2(movies_data)
    histogram_json5 = RNN(movies_data)
    histogram_json6 = RNN2(movies_data)
    return render_template('MLalgo.html', histogram_json=histogram_json, histogram_json2=histogram_json2,histogram_json3=histogram_json3,histogram_json4=histogram_json4,histogram_json5=histogram_json5,histogram_json6=histogram_json6)

