#from models import 
from flask import Flask, render_template, url_for, redirect, request,\
     send_from_directory
import os.path
#import time
#import ast

#### If our web app involves the user uploading files, we'll probably need these.

#from werkzeug.utils import secure_filename 
#ALLOWED_EXTENSIONS = set(['bmp', 'gif', 'jpg', 'jpeg', 'png'])
##file_dir = os.path.dirname(os.path.abspath(__file__))
#UPLOAD_FOLDER = os.path.join(file_dir, 'static/post_images')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)