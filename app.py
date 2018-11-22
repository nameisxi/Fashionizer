import os
import numpy as np
import cv2
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.mnist import MnistCreator

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_folder='css')

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

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def root():
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
            return redirect('https://fashionizer.herokuapp.com/results')
            #return redirect('http://localhost:5000/results')
    return send_from_directory('html', 'index.html')

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

@app.route('/results')
def results():
    return send_from_directory('html', 'results.html')

@app.route('/index-css')
def index_css():
    return send_from_directory('css', 'index.css')

@app.route('/upload-button-css')
def upload_button_css():
    return send_from_directory('css', 'upload-button.css')

if __name__ == '__main__':
    app.secret_key = 'something'
    app.run()