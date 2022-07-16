import ast
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpeg','png','webp']

@app.route('/send-caption', methods=['POST'])
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
        resp.status_code = 201
        #TODO: this is where we would do our model nonsense
        return resp
    else:
        resp = jsonify({'message' : 'File was not a valid image'})
        resp.status_code = 400
        return resp



if __name__ == "__main__":
    app.run()