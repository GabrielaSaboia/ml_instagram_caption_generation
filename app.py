import ast
import os
import asyncio
import time
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from backend.caption_pipeline import pipe

project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'frontend/templates')
app = Flask(__name__, template_folder=template_path)
FILE_PATH = 'uploaded_images/'
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpeg','png','webp']

@app.route('/')
def renderHomePage():
    return render_template('home.html')
@app.route('/get-caption-from-image', methods=['POST'])
def process():

    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']

    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        json = asyncio.run(pipe(file))
        resp = jsonify(json)
        resp.status_code = 201

        #TODO: this is where we would do our model nonsense
        return resp
    else:
        resp = jsonify({'message' : 'File was not a valid image'})
        resp.status_code = 400
        return resp



if __name__ == "__main__":
    app.run()