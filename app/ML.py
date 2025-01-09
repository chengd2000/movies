import json
import numpy as np
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Embedding
import os
import plotly.express as px
from app.utils import fetch_genres
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from flask import Flask, jsonify
from dotenv import load_dotenv

load_dotenv()  # טוען את המשתנים מהקובץ .env

API_KEY = os.getenv('API_KEY')

BASE_URL = "https://api.themoviedb.org/3"

def round_to_nearest_half(x):
    return round(x * 2) / 2

def diagram11(df):
    df=transform_data(df)
    movies_per_year = df['release_year'].value_counts().sort_index()
    fig = px.bar(
        x=movies_per_year.index,
        y=movies_per_year.values,
        labels={'x': 'Year', 'y': 'Number of Movies'},
        title="Number of Movies Per Year"
    )

    # עדכון הגרף לעיצוב טוב יותר
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Movies",
        autosize=True,
        template="plotly_white"
    )
    fig.write_html("app\static\Movies_By_Year.html")

    return fig.to_json()

def diagram12(df):
    df=transform_data(df)
    movies_per_month = df['release_month'].value_counts().sort_index()
    fig = px.bar(
        x=movies_per_month.index,
        y=movies_per_month.values,
        labels={'x': 'Month', 'y': 'Number of Movies'},
        title="Number of Movies Per Month"
    )

    # עדכון הגרף לעיצוב טוב יותר
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=[
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
        ),
        xaxis_title="Month",
        yaxis_title="Number of Movies",
        autosize=True,
        template="plotly_white"
    )

    fig.write_html("app\static\Movies_By_Month.html")

    return fig.to_json()

def diagram13(df):
    df=transform_data(df)
    df['rounded_vote_average']=df['vote_average'].apply(round_to_nearest_half)
    movies_per_vote_average = df['rounded_vote_average'].value_counts().sort_index()
    print(movies_per_vote_average)
    fig = px.bar(
        x=movies_per_vote_average.index,
        y=movies_per_vote_average.values,
        labels={'x': 'Vote Average', 'y': 'Number of Movies'},
        title="Number of Movies Per Vote Average"
    )

    # עדכון הגרף לעיצוב טוב יותר
    fig.update_layout(
        xaxis_title="Vote Average",
        yaxis_title="Number of Movies",
        autosize=True,
        template="plotly_white"
    )
    fig.write_html("app\static\Movies_Amount_By_Vote_Average.html")
    return fig.to_json()

def diagram14(df):
    df=transform_data(df)
    response = fetch_genres()
    genre_mapping_list = response["genres"]
    # Load genres_data into genre_mapping 
    genre_mapping = json.loads(json.dumps(genre_mapping_list)) #

    genres_list= []
    for genre in genre_mapping_list:
         genres_list.append(genre["name"])

    movies_per_genre = df[genres_list].sum()
    fig = px.pie(
        names=movies_per_genre.index,  # התוויות (שמות החודשים)
        values=movies_per_genre.values,  # הערכים (מספר הסרטים)
        title="Distribution of Movies by Genre",  # כותרת הגרף
        labels={"names": "Genre", "values": "Number of Movies"}
    )

    # עיצוב דיאגרמת ה-Pie
    fig.update_traces(textinfo='percent+label')  # הצגת אחוזים ושמות התוויות
    fig.update_layout(showlegend=False)
    fig.write_html("app\static\Movies_Amount_By_Genres.html")
    return fig.to_json()


def diagram15(df):
    df=transform_data(df)
    movies_per_lang = df['original_language'].value_counts().sort_index()
    fig = px.pie(
        names=movies_per_lang.index,  # התוויות (שמות החודשים)
        values=movies_per_lang.values,  # הערכים (מספר הסרטים)
        title="Distribution of Movies by Language",  # כותרת הגרף
        labels={"names": "Language", "values": "Number of Movies"}
    )

    # עיצוב דיאגרמת ה-Pie
    fig.update_traces(textinfo='none') # הצגת אחוזים ושמות התוויות
    fig.update_layout(
        template="plotly_white",
        legend_title="Language",
          # כותרת למקרא
    )
    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='none'
    )
    fig.write_html("app\static\Movies_Amount_By_Languages.html")
    return fig.to_json()


# פונקציה ליצירת ההיסטוגרמה באמצעות Plotly
def diagram(df):
    df=transform_data(df)
    # יצירת היסטוגרמה באמצעות Plotly
    fig = px.histogram(df, x="popularity", nbins=10, title="Histogram of Popularity", 
                       labels={"popularity": "Popularity"}, color_discrete_sequence=["blue"])
    fig.update_layout(xaxis_title="Popularity", yaxis_title="Frequency")
    # שמירת ההיסטוגרמה כקובץ HTML
    fig.write_html("app\static\popularity_histogram.html")
    # ניתן להחזיר את הגרף כ-HTML להטמעה באפליקציה
    return fig.to_json()

def diagram1(df):
    df=transform_data(df)
    # היסטוגרמה עבור vote_average
    fig = px.histogram(df, x="vote_average", nbins=10, title="Histogram of Vote Average", 
                       labels={"vote_average": "Vote Average"}, color_discrete_sequence=["green"])
    fig.update_layout(xaxis_title="Vote Average", yaxis_title="Frequency")
    fig.write_html("app\static\Vote_Average_histogram.html")
    return fig.to_json()

def diagram2(df):
    df=transform_data(df)
        # היסטוגרמה עבור vote_count
    fig = px.histogram(df, x="vote_count", nbins=10, title="Histogram of Vote Count", 
                       labels={"vote_count": "Vote Count"}, color_discrete_sequence=["red"])
    fig.update_layout(xaxis_title="Vote Count", yaxis_title="Frequency")
    fig.write_html("app\static\Vote_Count_histogram.html")
    return fig.to_json()

def diagram3(df):
    df=transform_data(df)
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df["popularity"],
        name="Popularity",
        marker_color="blue",
        boxmean="sd"  # הצגת ממוצע וסטיית תקן
    ))
    fig.update_layout(
        title="Box Plot of Popularity",
        xaxis_title="Popularity",
        yaxis=dict(visible=False)  # הסתרת ציר Y
    )
    fig.write_html("app\static\popularity_boxplot.html")
    return fig.to_json()

def diagram4(df):
    df=transform_data(df)
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df["vote_average"],
        name="Vote Average",
        marker_color="green",
        boxmean="sd"  # הצגת ממוצע וסטיית תקן
    ))
    fig.update_layout(
        title="Box Plot of Vote Average",
        xaxis_title="Vote Average",
        yaxis=dict(visible=False)  # הסתרת ציר Y
    )
    fig.write_html("app\static\Vote_Average_boxplot.html")
    return fig.to_json()

def diagram5(df):
    df=transform_data(df)
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df["vote_count"],
        name="Vote Count",
        marker_color="red",
        boxmean="sd"  # הצגת ממוצע וסטיית תקן
    ))
    fig.update_layout(
        title="Box Plot of Vote Count",
        xaxis_title="Vote Count",
        yaxis=dict(visible=False)  # הסתרת ציר Y
    )
    fig.write_html("app\static\Vote_Count_boxplot.html")
    return fig.to_json()

def diagram6(df):
    df=transform_data(df)
    fig = px.scatter(
        df,
        x="popularity",
        y="vote_count",
        title="Scatter Plot of Popularity vs Vote Count",
        labels={"popularity": "Popularity", "vote_count": "Vote Count"},
        color_discrete_sequence=["purple"]
    )
    fig.update_layout(
        xaxis_title="Popularity",
        yaxis_title="Vote Count"
    )
    fig.write_html("app\static\Popularity_VS_Vote_Count_scatter.html")
    return fig.to_json()

def diagram7(df):
    df=transform_data(df)
    fig = px.scatter(
        df,
        x="popularity",
        y="vote_average",
        title="Scatter Plot of Popularity vs Vote Average",
        labels={"popularity": "Popularity", "vote_average": "Vote Average"},
        color_discrete_sequence=["orange"]
    )
    fig.update_layout(
        xaxis_title="Popularity",
        yaxis_title="Vote Average"
    )
    fig.write_html("app\static\Popularity_VS_Vote_Average_scatter.html")
    return fig.to_json()

def diagram10(df):
    df=transform_data(df)
    fig = px.scatter(
        df,
        x="vote_count",
        y="vote_average",
        title="Scatter Plot of Vote Count vs Vote Average",
        labels={"vote_count": "Vote Count", "vote_average": "Vote Average"},
        color_discrete_sequence=["brown"]
    )
    fig.update_layout(
        xaxis_title="Vote Count",
        yaxis_title="Vote Average"
    )
    fig.write_html("app\static\Vote_Count_VS_Vote_Average_scatter.html")
    return fig.to_json()

def diagram8(df):
    df=df[['vote_count', 'vote_average', 'id','popularity']]
    fig = px.scatter_matrix(
        df,
        dimensions=df.columns,  # כל העמודות ב-DataFrame
        title="Pairplot ofpytho All Columns",
        labels={col: col.capitalize() for col in df.columns},  # תוויות לעמודות
        color_discrete_sequence=px.colors.qualitative.Pastel  # צבעים נעימים
    )
    fig.update_traces(diagonal_visible=False)  # הסתרת האלכסון
    fig.write_html("app\static\All_Columns_Matrix.html")
    return fig.to_json()

def diagram9(df):
    df=transform_data(df)
    df = df.drop(["video", "original_title", "backdrop_path", "poster_path","overview","original_language","adult","genre_ids","release_date","title"], axis=1)
    # חישוב מטריצת הקורלציות
    correlation_matrix = df.corr()

    # יצירת Heatmap באמצעות Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,  # ערכי הקורלציה
            x=correlation_matrix.columns,  # שמות העמודות לציר ה-X
            y=correlation_matrix.index,    # שמות העמודות לציר ה-Y
            colorscale='viridis',         # סקלת צבעים
            zmid=0,                        # מרכז הצבעים ב-0
            colorbar=dict(title="Correlation")  # כותרת לסרגל הצבעים
        )
    )

    # הגדרות עיצוב
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
        autosize=True
    )

    # החזרת הגרף בפורמט JSON
    fig.write_html("app\static\Correlation_Matrix.html")
    return fig.to_json()

def transform_data(df):
    # מחיקת עמודות לא נחוצות
    # if df["video"] and df["original_title"] and df["backdrop_path"] and df["poster_path"]:
    #     df = df.drop(["video", "original_title", "backdrop_path", "poster_path"], axis=1)
    # סינון סרטים עם פחות מ-200 הצבעות
    df = df[df["vote_count"] > 200]

    # מחיקת שורות עם ערכים חסרים
    df.dropna(inplace=True)

    # יצירת עמודות שנה וחודש
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year
    df['release_month'] = pd.to_datetime(df['release_date']).dt.month
    # Fetch the response
    response = fetch_genres()
    genre_mapping_list = response["genres"]
    # Load genres_data into genre_mapping 
    genre_mapping = json.loads(json.dumps(genre_mapping_list)) #

    genre_mapping = {genre["id"]: genre["name"] for genre in genre_mapping_list} 



        # ניצור עמודות לכל ז'אנר מתוך המיפוי
    for genre_id, genre_name in genre_mapping.items():
        # יוצרים עמודה חדשה לכל ז'אנר
        df[genre_name] = df['genre_ids'].apply(lambda x: 1 if genre_id in x else 0)
    sorted_df = df.sort_values(by="vote_average")
    return df




def fetch_all_pages(url, total_pages):
    """פונקציה לשליפת נתונים מכל העמודים"""
    all_data = []
    for page in range(1, total_pages + 1):
        print(f"משיכת נתונים מעמוד {page}...")
        response = requests.get(f"{url}&page={page}")
        
        if response.status_code == 200:
            data = response.json().get("results", [])
            
            df = pd.DataFrame(data)  # יצירת DataFrame
            response = fetch_genres()
            genre_mapping_list = response["genres"]
            genre_mapping = json.loads(json.dumps(genre_mapping_list)) #
            genre_mapping = {genre["id"]: genre["name"] for genre in genre_mapping_list}    # ניצור עמודות לכל ז'אנר מתוך המיפוי
            if(not df.empty):
                #if (df['genre_ids']):
                df['genre_ids'] = df['genre_ids'].apply(lambda x: [x] if not isinstance(x, (list, np.ndarray)) else x)
                for genre_id, genre_name in genre_mapping.items():
                    df[genre_name] = df['genre_ids'].apply(lambda x: 1 if genre_id in x else 0)
            #transformed_df = transform_data(df)  # עיבוד הנתונים
            all_data.append(df)
        else:
            print(f"שגיאה בשליפת נתונים מהעמוד {page}: {response.status_code}")
            break
    
    return pd.concat(all_data, ignore_index=True)  # איחוד כל הנתונים ל-DataFrame אחד





def perform_linear_regression(df):

    # שליפת כל הנתונים

    df=transform_data(df)
    
    # mlb = MultiLabelBinarizer()
    # genre_encoded = pd.DataFrame(mlb.fit_transform(df['genre_ids']), columns=mlb.classes)

    # אתחול תכונות ומטרה     + list(genre_encoded.columns)
    X = df[['vote_average', 'release_year', 'release_month'] ]
    y = df['popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # יצירת מודל רגרסיה
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ביצוע תחזיות
    y_pred = model.predict(X_test)

        # גרף מספר סרטים לכל שנה
    # movies_per_year = df['release_year'].value_counts().sort_index()
    # movies_per_year.plot(kind='bar', figsize=(10, 6), title="Number of Movies Per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Number of Movies")
    # plt.show()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    fig = go.Figure()

    # נקודות אמתיות מול תחזיות
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        marker=dict(color='blue', size=6),
        name='Predicted vs Actual'
    ))

    # קו אידיאלי (y = x)
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        line=dict(color='red', dash='dash'),
        name='Ideal Fit'
    ))

    # עיצוב הגרף
    fig.update_layout(
        title="Linear Regression Predictions vs Actual Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    fig.write_html("app\static\Linear_Regression.html")

    return fig.to_json()
    
    # return list(y_pred)  # החזרת התוצאות

# @main.route('/fetch_data_for_algorithm', methods=['GET'])
# def fetch_data_for_algorithm():
#     movies_data = fetch_movies()  # שליפת נתונים
#     predictions = perform_linear_regression(movies_data)  # ביצוע האלגוריתם
#     return jsonify(predictions)  # החזרת התוצאות לקוח

# def k_means(data):

# # שליפת כל הנתונים
#     total_pages = 100  # מספר העמודים שברצונך למשוך (אפשר גם לחשב מתוך ה-API אם יש מידע)

#     df = fetch_all_pages(f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}", total_pages)
    
#     # בחר את העמודות שאתה רוצה להשתמש בהן
#     features = ['popularity', 'vote_average', 'vote_count']
#     X = df[features]

#     # נורמליזציה של הנתונים (כדי להבטיח שכל הפיצ'רים יהיו באותו טווח)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # הגדרת מספר הקבוצות
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     kmeans.fit(X_scaled)

#     # הוספת עמודת הקבוצה לכל סרט
#     df['cluster'] = kmeans.labels_

#         # חישוב אלבוא
#     distortions = []
#     for k in range(1, 11):
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(X_scaled)
#         distortions.append(kmeans.inertia_)

#     plt.plot(range(1, 11), distortions, marker='o')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Distortion')
#     plt.title('Elbow Method')
#     plt.show()


#     @main.route('/fetch_data_for_algorithm', methods=['GET'])
#     def fetch_data_for_algorithm():
#         movies_data = fetch_movies()  # שליפת נתונים
#         predictions = k_means(movies_data)  # ביצוע האלגוריתם
#         return jsonify(predictions)  # החזרת התוצאות לקוח



app = Flask(__name__)

# פונקציה ל-K-Means
def k_means(df):
    df=transform_data(df)
    # בחירת עמודות פיצ'רים
    features = ['popularity', 'vote_average', 'vote_count']
    X = df[features]

    # נורמליזציה של הנתונים
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # הפעלת KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    # הוספת עמודת קבוצות
    df['cluster'] = kmeans.labels_

    # חישוב שיטת המרפק
    distortions = []
    for k in range(1, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        distortions.append(kmeans_temp.inertia_)

    # גרף שיטת המרפק
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=distortions,
        mode='lines+markers',
        name='Distortion'
    ))
    elbow_fig.update_layout(
        title='K-Means Elbow Method',
        xaxis_title='Number of Clusters',
        yaxis_title='Distortion'
    )

    # # גרף פיזור תלת-ממדי לפי קבוצות
    # scatter_fig = px.scatter_3d(
    #     df,
    #     x='popularity',
    #     y='vote_average',
    #     z='vote_count',
    #     color='cluster',
    #     title="K-Means Clustering Results",
    #     labels={'cluster': 'Cluster'}
    # )

    elbow_fig.write_html("app\static\k_means.html")

    return df, elbow_fig.to_json()


def k_means2(df):
    df=transform_data(df)
    # בחירת עמודות פיצ'רים
    features = ['popularity', 'vote_average', 'vote_count']
    X = df[features]

    # נורמליזציה של הנתונים
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # הפעלת KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    # הוספת עמודת קבוצות
    df['cluster'] = kmeans.labels_

    # חישוב שיטת המרפק
    distortions = []
    for k in range(1, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        distortions.append(kmeans_temp.inertia_)

    # # גרף שיטת המרפק
    # elbow_fig = go.Figure()
    # elbow_fig.add_trace(go.Scatter(
    #     x=list(range(1, 11)),
    #     y=distortions,
    #     mode='lines+markers',
    #     name='Distortion'
    # ))
    # elbow_fig.update_layout(
    #     title='Elbow Method',
    #     xaxis_title='Number of Clusters',
    #     yaxis_title='Distortion'
    # )

    # גרף פיזור תלת-ממדי לפי קבוצות
    scatter_fig = px.scatter_3d(
        df,
        x='popularity',
        y='vote_average',
        z='vote_count',
        color='cluster',
        title="K-Means Clustering Results",
        labels={'cluster': 'Cluster'}
    )

    scatter_fig.write_html("app\static\k_means2.html")

    return df, scatter_fig.to_json()




# def k_means(data):
#     # בחר את העמודות שבהן משתמשים
#     features = ['popularity', 'vote_average', 'vote_count']
#     X = data[features]

#     # נורמליזציה של הנתונים
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # הגדרת מספר הקבוצות
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     kmeans.fit(X_scaled)

#     # הוספת עמודת הקבוצה לכל סרט
#     data['cluster'] = kmeans.labels_

#     # חישוב שיטת המרפק
#     distortions = []
#     for k in range(1, 11):
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(X_scaled)
#         distortions.append(kmeans.inertia_)

#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, 11), distortions, marker='o')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Distortion')
#     plt.title('Elbow Method')
#     plt.show()

#     return data


# @main.route('/fetch_data_for_algorithm', methods=['GET'])
# def fetch_data_for_algorithm():
#     total_pages = 100  # מספר העמודים שברצונך למשוך
#     url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"
#     movies_data = fetch_all_pages(url, total_pages)  # שליפת נתונים
#     predictions = k_means(movies_data)  # ביצוע האלגוריתם
#     return jsonify(predictions.to_dict(orient='records'))  # החזרת התוצאות



def RandomForest(df):
    # Transform the data
    df = transform_data(df)
    
    # יצירת עמודת יעד (target) לפי הפופולריות
    threshold = 100
    df['is_popular'] = np.where(df['popularity'] >= threshold, 1, 0)
    
    # בחירת פיצ'רים
    features = ['vote_average']
    X = df[features]
    y = df['is_popular']

    # חלוקה לסטי אימון ובדיקה
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # יצירת המודל
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # בדיקה אם התיקייה קיימת, ואם לא - ליצור אותה
    output_dir = "app/static/random_forest/"
    os.makedirs(output_dir, exist_ok=True)

    figs = []
    for i, estimator in enumerate(rf_model.estimators_):
        tree_text = export_text(estimator, feature_names=features)
        print(tree_text)
        lines = tree_text.split("\n")
        tree_dict = {"levels": [], "features": []}

        for line in lines:
            level = line.count("|")
            content = line.replace("|", "").strip()
            if content:
                tree_dict["levels"].append(level)
                tree_dict["features"].append(content)
        print(tree_dict)

        # יצירת מבנה היררכי
        parents = []
        for j, level in enumerate(tree_dict["levels"]):
            if level == 0:
                parents.append("")
            else:
                parent_index = next((k for k in range(j-1, -1, -1) if tree_dict["levels"][k] == level-1), None)
                parents.append(tree_dict["features"][parent_index] if parent_index is not None else "")

        # יצירת גרף Sunburst
        fig = go.Figure(go.Sunburst(
            labels=tree_dict["features"],
            parents=parents,
            maxdepth=3,
            branchvalues="total"
        ))
        fig.update_layout(title=f"Random Forest Decision Tree (Estimator {i + 1})")
        figs.append(fig)
        fig.write_html(f"{output_dir}Random_Forest_{i + 1}.html")

    # מחזירים את ה-JSON של העץ הראשון
    return figs[0].to_json()


# def RandomForest(df):

#     df = transform_data(df)
    
#     # יצירת עמודת יעד (target) לפי הפופולריות
#     threshold = 100  # ערך הסף להחלטה
#     df['is_popular'] = np.where(df['popularity'] >= threshold, 1, 0)
    
#     # בחירת פיצ'רים
#     features = ['vote_average']  # אפשר להרחיב בהתאם לצורך
#     X = df[features]

#     # עמודת היעד
#     y = df['is_popular']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # יצירת המודל
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#     # אימון המודל
#     rf_model.fit(X_train, y_train)

#     # ניבוי על סט הבדיקה
#     y_pred = rf_model.predict(X_test)

#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))

#     # Create Plotly figures for multiple decision trees
#     figs = []
#     for i in range(len(rf_model.estimators_)):  # Adjust the range for more/less trees
#         tree_text = export_text(rf_model.estimators_[i], feature_names=features)
#         lines = tree_text.split("\n")
#         tree_dict = {"levels": [], "features": []}

#         for line in lines:
#             level = line.count("|")
#             content = line.replace("|", "").strip()
#             if content:
#                 tree_dict["levels"].append(level)
#                 tree_dict["features"].append(content)

#         # Fix to ensure parent exists
#         parents = []
#         for j, level in enumerate(tree_dict["levels"]):
#             if level == 0:
#                 parents.append("")
#             else:
#                 parent_level = level - 1
#                 parent_index = None
#                 for k in range(j-1, -1, -1):
#                     if tree_dict["levels"][k] == parent_level:
#                         parent_index = k
#                         break
#                 if parent_index is None:
#                     parents.append("")
#                 else:
#                     parents.append(tree_dict["features"][parent_index])

#         # יצירת גרף Plotly עבור כל עץ
#         fig = go.Figure(go.Sunburst(
#             labels=tree_dict["features"],
#             parents=parents,
#             maxdepth=3,  # עומק העץ להצגה
#             branchvalues="total"
#         ))
#         fig.update_layout(title=f"Random Forest Decision Tree (Estimator {i + 1})")
#         figs.append(fig)

#     # Save figures to HTML files
#     for i, fig in enumerate(figs):
#         fig.write_html(f"app/static/random_forest/Random_Forest_{i + 1}.html")

#     # Return the first figure's JSON as an example (modify as needed)
#     return figs[0].to_json()


    # df=transform_data(df)
    # # יצירת עמודת יעד (target) לפי הפופולריות
    #  # יצירת עמודת יעד (target) לפי פופולריות
    # threshold = 100  # ערך סף
    # df['is_popular'] = np.where(df['popularity'] >= threshold, 1, 0)

    # # בחירת פיצ'רים
    # features = ['vote_average']  # ניתן להוסיף פיצ'רים נוספים
    # X = df[features]
    # y = df['is_popular']

    # # פיצול לנתוני אימון ובדיקה
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # יצירת המודל ואימון
    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_model.fit(X_train, y_train)
    
    # # הצגת עץ ההחלטה הראשון
    # estimator = rf_model.estimators_[0]
    # tree_text = export_text(estimator, feature_names=features)
    # print(tree_text)
    # # המרת עץ ההחלטה לפורמט מתאים ל-Plotly
    # lines = tree_text.split("\n")
    # tree_dict = {"levels": [], "features": []}
    # for line in lines:
    #     level = line.count("|")
    #     content = line.replace("|", "").strip()
    #     if content:
    #         tree_dict["levels"].append(level)
    #         tree_dict["features"].append(content)
    # parents = [] 
    # for j, level in enumerate(tree_dict["levels"]): 
    #     if level == 0: parents.append("") 
    #     else: 
    #         try: 
    #             parents.append(tree_dict["features"][tree_dict["levels"][:j].index(level - 1)]) 
    #         except ValueError: parents.append("")
    # # יצירת גרף Plotly
    # # יצירת גרף Plotly עבור כל עץ
    # fig = go.Figure(maxdepth=3, branchvalues="total" )
    # fig.update_layout(title=f"Random Forest Decision Tree")    
    #     # הצגת העץ הראשון מתוך היער
    # # estimator = rf_model.estimators_[0]  # בחר את העץ הראשון מתוך היער

    # # # הצגת העץ
    # # plt.figure(figsize=(20, 10))
    # # plot_tree(estimator, filled=True, feature_names=features, class_names=['Not Popular', 'Popular'], rounded=True)
    # # plt.show()
    


    # fig.write_html("app\static\Random_Forest.html")

    # return fig.to_json()


# @main.route('/fetch_data_for_algorithm', methods=['GET'])
# def fetch_data_for_algorithm():
#     total_pages = 100  # מספר העמודים שברצונך למשוך
#     url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"
#     movies_data = fetch_all_pages(url, total_pages)  # שליפת נתונים
#     predictions = RandomForest(movies_data)  # ביצוע האלגוריתם
#     return jsonify(predictions.to_dict(orient='records'))  # החזרת התוצאות




import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import plotly.graph_objects as go
from flask import Flask, jsonify

app = Flask(__name__)

def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i + seq_length])
        y_seq.append(data[i + seq_length][0])  # נניח שהיעד הוא הפופולריות
    return np.array(X_seq), np.array(y_seq)

def RNN(df):
    df = transform_data(df)
    # מיון לפי תאריך יציאה
    df = df.sort_values('release_date')

    # בחירת פיצ'רים
    features = ['popularity', 'vote_average', 'vote_count']
    X = df[features].values

    # נורמליזציה
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # יצירת רצפים
    sequence_length = 10  # חלון זמן של 10
    X_seq, y_seq = create_sequences(X_scaled, sequence_length)

    # בניית המודל
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(sequence_length, X_seq.shape[2])),
        Dense(1)  # שכבת פלט לחיזוי
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # חלוקת הנתונים
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # אימון המודל
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=0)

    # הערכת המודל
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss}, MAE: {mae}")

    # גרף אובדן (Loss)
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
    loss_fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    loss_fig.update_layout(
        title="RNN - Training vs Validation Loss",
        xaxis_title="Epochs",
        yaxis_title="Loss"
    )

    # # תחזיות
    # y_pred = model.predict(X_test, verbose=0)

    # # גרף חיזוי מול תוצאות אמיתיות
    # prediction_fig = go.Figure()
    # prediction_fig.add_trace(go.Scatter(y=y_test[:50], mode='lines+markers', name='Actual'))
    # prediction_fig.add_trace(go.Scatter(y=y_pred[:50].flatten(), mode='lines+markers', name='Predicted'))
    # prediction_fig.update_layout(
    #     title="Actual vs Predicted (First 50 Samples)",
    #     xaxis_title="Sample Index",
    #     yaxis_title="Popularity"
    # )

    loss_fig.write_html("app\static\RNN.html")

    return df, loss_fig.to_json()


def RNN2(df):
    df = transform_data(df)
    # מיון לפי תאריך יציאה
    df = df.sort_values('release_date')

    # בחירת פיצ'רים
    features = ['popularity', 'vote_average', 'vote_count']
    X = df[features].values

    # נורמליזציה
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # יצירת רצפים
    sequence_length = 10  # חלון זמן של 10
    X_seq, y_seq = create_sequences(X_scaled, sequence_length)

    # בניית המודל
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(sequence_length, X_seq.shape[2])),
        Dense(1)  # שכבת פלט לחיזוי
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # חלוקת הנתונים
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # אימון המודל
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=0)

    # # הערכת המודל
    # loss, mae = model.evaluate(X_test, y_test, verbose=0)
    # print(f"Loss: {loss}, MAE: {mae}")

    # # גרף אובדן (Loss)
    # loss_fig = go.Figure()
    # loss_fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
    # loss_fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    # loss_fig.update_layout(
    #     title="Training vs Validation Loss",
    #     xaxis_title="Epochs",
    #     yaxis_title="Loss"
    # )

    # תחזיות
    y_pred = model.predict(X_test, verbose=0)

    # גרף חיזוי מול תוצאות אמיתיות
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(y=y_test[:50], mode='lines+markers', name='Actual'))
    prediction_fig.add_trace(go.Scatter(y=y_pred[:50].flatten(), mode='lines+markers', name='Predicted'))
    prediction_fig.update_layout(
        title="RNN - Actual vs Predicted (First 50 Samples)",
        xaxis_title="Sample Index",
        yaxis_title="Popularity"
    )

    prediction_fig.write_html("app\static\RNN2.html")

    return df, prediction_fig.to_json()


# def create_sequences(data, seq_length):
#         X_seq, y_seq = [], []
#         for i in range(len(data) - seq_length):
#             X_seq.append(data[i:i + seq_length])
#             y_seq.append(data[i + seq_length][0])  # נניח שהיעד הוא הפופולריות
#         return np.array(X_seq), np.array(y_seq)

# def RNN(df):
#     df = df.sort_values('release_date')

#     features = ['popularity', 'vote_average', 'vote_count']
#     X = df[features].values

#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)


#     sequence_length = 10  # חלון זמן של 10
#     X_seq, y_seq = create_sequences(X_scaled, sequence_length)



#     model = Sequential([
#         SimpleRNN(50, activation='relu', input_shape=(sequence_length, X_seq.shape[2])),
#         Dense(1)  # שכבה פלט לחיזוי הפופולריות
#     ])

#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     model.summary()


#     X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

#     # אימון
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)


#     # הערכה
#     loss, mae = model.evaluate(X_test, y_test)
#     print(f"Loss: {loss}, MAE: {mae}")

#     # # גרף אובדן
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.legend()
#     plt.show()



#     # חיזוי
#     y_pred = model.predict(X_test)

#     # השוואה בין תוצאות
#     plt.plot(y_test[:50], label='Actual')
#     plt.plot(y_pred[:50], label='Predicted')
#     plt.legend()
#     plt.show()


#     return df



# @main.route('/fetch_data_for_algorithm', methods=['GET'])
# def fetch_data_for_algorithm():
#     total_pages = 100  # מספר העמודים שברצונך למשוך
#     url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"
#     movies_data = fetch_all_pages(url, total_pages)  # שליפת נתונים
#     predictions = RNN(movies_data)  # ביצוע האלגוריתם
#     return jsonify(predictions.to_dict(orient='records'))  # החזרת התוצאות


# def test(df):
#     # היסטוגרמה עבור popularity
#     plt.hist(df["popularity"], bins=10, color='blue', edgecolor='black')
#     plt.title("Histogram of Popularity")
#     plt.xlabel("Popularity")
#     plt.ylabel("Frequency")
#     plt.show()

#     return df

# def test1(df):
#     # היסטוגרמה עבור vote_average
#     plt.hist(df["vote_average"], bins=10, color='green', edgecolor='black')
#     plt.title("Histogram of Vote Average")
#     plt.xlabel("Vote Average")
#     plt.ylabel("Frequency")
#     plt.show()
#     return df


# def test2(df):
#         # היסטוגרמה עבור vote_count
#     plt.hist(df["vote_count"], bins=10, color='red', edgecolor='black')
#     plt.title("Histogram of Vote Count")
#     plt.xlabel("Vote Count")
#     plt.ylabel("Frequency")
#     plt.show()

#     return df


# def test3(df):
#     # Box plot עבור popularity
#     df["popularity"].plot(kind='box', vert=False, color='blue')
#     plt.title("Box Plot of Popularity")
#     plt.show()

#     return df


# def test4(df):
#     # Box plot עבור vote_average
#     df["vote_average"].plot(kind='box', vert=False, color='green')
#     plt.title("Box Plot of Vote Average")
#     plt.show()

#     return df


# def test5(df):
#     # Box plot עבור vote_count
#     df["vote_count"].plot(kind='box', vert=False, color='red')
#     plt.title("Box Plot of Vote Count")
#     plt.show()

#     return df


# def test6(df):
#     # Scatter plot בין popularity ו-vote_count
#     plt.scatter(df["popularity"], df["vote_count"], color='purple')
#     plt.title("Scatter Plot of Popularity vs Vote Count")
#     plt.xlabel("Popularity")
#     plt.ylabel("Vote Count")
#     plt.show()

#     return df


# def test7(df):
#     # Scatter plot בין popularity ו-vote_average
#     plt.scatter(df["popularity"], df["vote_average"], color='orange')
#     plt.title("Scatter Plot of Popularity vs Vote Average")
#     plt.xlabel("Popularity")
#     plt.ylabel("Vote Average")
#     plt.show()

#     return df

#לבדוק מה לעשות לגבי זה
# def test8(df):
#     # # df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x))

#     # # Pairplot בין כל העמודות
#     # sns.pairplot(df)
#     # plt.show()

#     # return df
#      # שמירה רק על עמודות מספריות
#     df_numeric = df.select_dtypes(include=['number'])
    
#     # בדיקה אם יש עמודות בעלות ערך זהה לכל השורות (עמודות לא רלוונטיות)
#     df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]
    
#     # יצירת Pairplot בין העמודות הנבחרות
#     sns.pairplot(df_numeric)
#     plt.show()

#     return df_numeric


# def test9(df):
#     # חישוב קורלציה
#     correlation_matrix = df.corr()

#     # הצגת קורלציה בעזרת Heatmap
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#     plt.title("Correlation Heatmap")
#     plt.show()

#     return correlation_matrix


# def test10(df):
#     # הפקודה plt.scatter
#     plt.scatter(x = df["vote_count"], y = df["vote_average"])

#     # הוספת כותרת ותיוג צירים
#     plt.title("vote_count vs vote_average")
#     plt.xlabel('vote_count')
#     plt.ylabel('vote_average')

#     # הצגת הגרף
#     plt.show()

#     return df





# @main.route('/fetch_data_for_algorithm', methods=['GET'])
# def fetch_data_for_algorithm():
#     total_pages = 100  # מספר העמודים שברצונך למשוך
#     url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"
#     movies_data = fetch_all_pages(url, total_pages)  # שליפת נתונים
#     predictions = test8(movies_data)  # ביצוע האלגוריתם
#     return jsonify(predictions.to_dict(orient='records'))  # החזרת התוצאות


# def test9(df):
#     # סינון עמודות מספריות בלבד
#     numeric_columns = df.select_dtypes(include=['number'])
    
#     # בדיקה אם יש עמודות מספריות
#     if numeric_columns.empty:
#         raise ValueError("The dataset does not contain any numeric columns for correlation analysis.")
    
#     # חישוב קורלציה
#     correlation_matrix = numeric_columns.corr()

#     # הצגת קורלציה בעזרת Heatmap
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#     plt.title("Correlation Heatmap")
#     plt.show()

#     return correlation_matrix

# @main.route('/fetch_data_for_algorithm', methods=['GET'])
# def fetch_data_for_algorithm():
#     total_pages = 100  # מספר העמודים שברצונך למשוך
#     url = f"{BASE_URL}/movie/popular?language=en-US&api_key={API_KEY}"
#     movies_data = fetch_all_pages(url, total_pages)  # שליפת נתונים
    
#     # הפיכת הנתונים ל-DataFrame
#     movies_df = pd.DataFrame(movies_data)
    
#     # ביצוע האלגוריתם
#     predictions = test9(movies_df)
    
#     return jsonify(predictions.to_dict(orient='records'))  # החזרת התוצאות
















