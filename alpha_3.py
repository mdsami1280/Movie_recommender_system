import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

app = Flask(__name__)

# Load the NLP model and TFIDF vectorizer from disk
with open('nlp_model.pkl', 'rb') as file:
    clf = pickle.load(file)
with open('tranform.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Global variables to store data and similarity matrix
data = None
similarity = None

def create_similarity():
    global data, similarity
    data = pd.read_csv('main_data.csv')
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)

def rcmd(m):
    global data, similarity
    m = m.lower()
    if data is None or similarity is None:
        create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies.'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Exclude the first item since it is the requested movie itself
        recommended_movies = [data['movie_title'][idx] for idx, _ in lst]
        return recommended_movies

def convert_to_list(my_list):
    my_list = my_list.strip(']["').split('","')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form['name']
    rc = rcmd(movie)
    if isinstance(rc, str):
        return rc
    else:
        return "---".join(rc)

@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Convert strings to lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    cast_ids = cast_ids.strip('[]').split(',')

    # Clean cast bios
    cast_bios = [bio.replace(r'\n', '\n').replace(r'\"', '\"') for bio in cast_bios]

    # Combine lists into dictionaries
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping to get user reviews from IMDb
    sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []
    reviews_status = []
    for review in soup_result:
        if review.string:
            reviews_list.append(review.string)
            movie_review_list = np.array([review.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    # Render template with all data
    return render_template(
        'recommend.html',
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details
    )

if __name__ == '__main__':
    app.run(debug=True)
