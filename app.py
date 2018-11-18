import os
import numpy as np
from PIL import Image
import cv2
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.mnist import MnistCreator

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pre_process_image(image, filename):
    mnist_creator = MnistCreator()
    image_data = np.asarray(image)
    image_data = mnist_creator.crop(image_data)
    image_data = mnist_creator.trim(image_data)
    img = Image.fromarray(image_data.astype('uint8'), 'RGB')
    img.save('result.png')
    image = mnist_creator.resize_longest_edge('result.png')
    image_data = np.asarray(image)
    image_data = mnist_creator.extend_shortest_edge(image, image_data)
    negative = mnist_creator.negate_intensities(image_data)
    grayscale = mnist_creator.convert_to_grayscale(negative)
    cv2.imwrite('result.png', grayscale)

@app.route('/', methods=['GET', 'POST'])
def root():
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
            return redirect('http://localhost:5000/results')
    return send_from_directory('html', 'index.html')

@app.route('/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload')
def upload():
    print("upload nauttais toimivan")
    '''if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("file saving toimii")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pre_process_image(file)
            return send_from_directory('html', 'upload.html')
    return redirect('http://localhost:5000/')'''
    #return send_from_directory('html', 'upload.html')
    return redirect('http://localhost:5000/results')

@app.route('/results')
def results():
    return send_from_directory('html', 'results.html')

if __name__ == '__main__':
    app.secret_key = 'something'
    app.run(debug = True)