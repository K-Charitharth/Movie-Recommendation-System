from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np

app = Flask(__name__)

# Load data
movies = pd.read_csv('D:/ML/projects/Movie-Recommendation-System/jupyter_notebook/Data/movies.csv')
links = pd.read_csv('D:/ML/projects/Movie-Recommendation-System/jupyter_notebook/Data/links.csv')  # Contains movieId and tmdbId

with open('artifacts/movie_inverse_mapper.pkl', 'rb') as f:
    movie_inv_mapper = pickle.load(f)

with open('artifacts/movie_mapper.pkl', 'rb') as f:
    movie_mapper = pickle.load(f)

with open('artifacts/utility_matrix.pkl', 'rb') as f:
    X = pickle.load(f)

with open('artifacts/model.pkl', 'rb') as f:
    kNN = pickle.load(f)


def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]



def get_recommendations(title, X):
    X = X.T
    idx = movie_mapper[dict(zip(movies['title'], movies['movieId']))[title]]
    movie_vec = X[idx]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    
    neighbours = kNN.kneighbors(movie_vec, return_distance=False).flatten()
    similar_movies = [movie_inv_mapper[n] for n in neighbours if n != idx]
    
    recommendations = movies.iloc[similar_movies][['movieId', 'title']]
    recommendations['link'] = recommendations['movieId'].map(
        lambda x: f"https://www.themoviedb.org/movie/{links.loc[links['movieId'] == x, 'tmdbId'].values[0]}"
    )
    print(recommendations)
    return recommendations

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title')
    recommendations = get_recommendations(movie_title, X)
    print(recommendations)
    return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    if len(query) < 3:
        return jsonify([])

    matches = process.extractBests(query, movies['title'].tolist(), limit=10)
    return jsonify([match[0] for match in matches])




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
