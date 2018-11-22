import os
import numpy as np
import cv2
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.mnist import MnistCreator

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

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
    #return send_from_directory('./tmp', 'result.png')
    return send_from_directory('tmp', 'result.png')

@app.route('/model')
def get_model():
    return send_from_directory('model', 'model.json')

@app.route('/group1-shard1of1')
def get_shard():
    return send_from_directory('model', 'group1-shard1of1')

@app.route('/results')
def results():
    #response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    #response.headers["Pragma"] = "no-cache"
    #response.headers["Expires"] = "0"
    return send_from_directory('html', 'results.html')

if __name__ == '__main__':
    app.secret_key = 'something'
    app.run()