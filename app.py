import os
import numpy as np
from PIL import Image
import cv2
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.mnist import MnistCreator

#UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pre_process_image(image, filename):
    mnist_creator = MnistCreator()
    image = mnist_creator.get_as_png('./tmp/' + filename, './tmp/img.png')
    image_data = np.asarray(image)
    image_data = mnist_creator.crop(image_data)
    image_data = mnist_creator.trim(image_data)
    img = Image.fromarray(image_data.astype('uint8'), 'RGB')
    img.save('./tmp/result.png')
    image = mnist_creator.resize_longest_edge('result.png')
    image_data = np.asarray(image)
    image_data = mnist_creator.extend_shortest_edge(image, image_data)
    negative = mnist_creator.negate_intensities(image_data)
    grayscale = mnist_creator.convert_to_grayscale(negative)
    cv2.imwrite('./tmp/result.png', grayscale)

@app.route('/', methods=['GET', 'POST'])
def root():
    if os.path.exists("./tmp/result.png"):
        os.remove("./tmp/result.png")
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
            pre_process_image(file, filename)
            return redirect('https://fashionizer.herokuapp.com/results')
    return send_from_directory('html', 'index.html')

@app.route('/img')
def get_img():
    return send_from_directory('tmp', 'result.png')

@app.route('/model')
def get_model():
    return send_from_directory('model', 'model.json')

@app.route('/group1-shard1of1')
def get_shard():
    return send_from_directory('model', 'group1-shard1of1')

@app.route('/results')
def results():
    if os.path.exists("./tmp/img.png"):
        os.remove("./tmp/img.png")
    return send_from_directory('html', 'results.html')

if __name__ == '__main__':
    app.secret_key = 'something'
    app.run(debug = True)