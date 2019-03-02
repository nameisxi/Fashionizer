import os
import numpy as np
import cv2
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import json
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.mnist import MnistCreator

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_folder='css')

input_vector = None

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def root():
    return send_from_directory('html', 'index.html')


#------------------------------------------------------------------#
# Apparel classification                                           #
#------------------------------------------------------------------#
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pre_process_image(image, filename):
    mnist_creator = MnistCreator()
    path_to_original = './tmp/' + filename
    path = './tmp/img.png'
    image = mnist_creator.get_as_png(path_to_original, path)
    image_data = mnist_creator.remove_background(image)
    image_data = mnist_creator.trim(image_data)
    image_data = mnist_creator.resize_longest_edge(path, image_data)
    image_data = mnist_creator.extend_shortest_edge(path, image_data)
    image_data = mnist_creator.negate_intensities(image_data)
    image = mnist_creator.convert_to_grayscale(path, image_data)
    cv2.imwrite('./tmp/result.png', image)
    
    if os.path.exists(".tmp/img.png"):
        os.remove(".tmp/img.png")

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if os.path.exists("./tmp/result.png"):
        os.rename("./tmp/result.png", "./tmp/old.png")
        os.remove("./tmp/old.png")

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename('img.png')
            file.save(os.path.join(app.root_path, 'tmp', filename))
            #file.save(os.path.join("/tmp/", filename))
            pre_process_image(file, filename)
            #return redirect('https://fashionizer.herokuapp.com/classification-results')
            return redirect('http://localhost:5000/classification-results')
    return send_from_directory('html', 'classify.html')

@app.route('/img')
def get_img():
    return send_from_directory('tmp', 'result.png')

@app.route('/original-img')
def get_original_img():
    return send_from_directory('tmp', 'img.png')

@app.route('/model')
def get_model():
    return send_from_directory('model', 'model.json')

@app.route('/group1-shard1of1')
def get_shard():
    return send_from_directory('model', 'group1-shard1of1')

@app.route('/classification-results')
def classification_results():
    return send_from_directory('html', 'classification-results.html')


#------------------------------------------------------------------#
# Apparel recommendation                                           #
#------------------------------------------------------------------#
'''def get_artists(sp, uri):
    username = uri.split(':')[2]
    playlist_id = uri.split(':')[4]

    results = sp.user_playlist(username, playlist_id)
    tracks = results['tracks']

    artists = []
    for item in tracks['items']:
        if 'track' in item:
            track = item['track']
        else:
            track = item
        try:
            track_url = track['external_urls']['spotify']
            name = track['artists'][0]['name']
            artists.append(name)
        except KeyError:
            print('Skipping track')
    return artists

def get_genres(sp, uri):
    client_credentials_manager = SpotifyClientCredentials(client_id='cd41a8e9db1d4b6dbb646d287a31b85a', client_secret='31c35bcfa1da4b84afaaf15e6987aa0e')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    artists = get_artists(uri)

    genres =  []
    for artist in artists:
    result = sp.search(q='artist:' + artist, type='artist')
    try:
        genre = result['artists']['items'][0]['genres']
        if len(genre) > 1:
            for g in genre:
                genres.append(g)
        else:
            genres.append(genre[0])
    except IndexError:
        print("Skipping artist")
    return genres

def get_input_vector(genres):
    input_vector = np.zeros((10, 1))
    for genre in genres:
        genre = genre.strip().lower()
        if 'pop' in genre or 'r&b' in genre:
            input_vector[0, 0] += 1
        elif 'rap' in genre:
            input_vector[1, 0] += 1
        elif 'classical' in genre:
            input_vector[2, 0] += 1
        elif 'metal' in genre or 'heavy' in genre:
            input_vector[3, 0] += 1
        elif 'rock' in genre:
            input_vector[4, 0] += 1
        elif 'hip-hop' in genre or 'chill' in genre or 'lofi' in genre:
            input_vector[5, 0] += 1
        elif 'jazz' in genre:
            input_vector[6, 0] += 1
        elif 'blues' in genre:
            input_vector[7, 0] += 1
        elif 'electronic' in genre or 'electro' in genre or 'edm' in genre or 'trap' in genre:
            input_vector[8, 0] += 1
        else:
            input_vector[9, 0] += 1
    return input_vector

def get_predictions(X):
    return None

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    print("toimii")
    if request.method == 'POST':
        if 'text' not in request.form:
            flash('No input was given')
            return redirect(request.url) 
        playlist_id = request.form['text']
        uri = 'spotify:user:spotifycharts:playlist:' + playlist_id
        genres = get_genres(uri)
        input_vector = get_input_vector(genres)
        #return redirect('https://fashionizer.herokuapp.com/recommendation-results')
        return redirect('http://localhost:5000/recommendation-results')
    return send_from_directory('html', 'recommend.html')

@app.route('/recommendation-results')
def recommendation_results():
    if input_vector == None:
        #return redirect('https://fashionizer.herokuapp.com/recommend')
        return redirect('http://localhost:5000/recommend')
    predictions = get_predictions(input_vector)
    #return send_from_directory('html', 'recommendation-results.html')
    return render_template('recommendation-results.html', prediction=predictions)'''


#------------------------------------------------------------------#
# CSS                                                              #
#------------------------------------------------------------------#
@app.route('/index-css')
def index_css():
    return send_from_directory('css', 'index.css')

@app.route('/upload-button-css')
def upload_button_css():
    return send_from_directory('css', 'upload-button.css')

if __name__ == '__main__':
    app.secret_key = 'something'
    app.run()