#from models import 
from flask import Flask, render_template, url_for, redirect, request,\
     send_from_directory
import os.path
import time
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

@app.route('/scamtextdet')
def ScamTextDetect():
    #time.sleep(10)
    return render_template('scamtextdetector.html')

@app.route('/scamcalldet')
def ScamCallDetect():
    return render_template('scamcalldetector.html')

#@app.route('/loading')
#def loading():
    #return render_template("loading.html")

@app.route('/output', methods=["POST"])
def ScamTextInput():
    time.sleep(10)
    print("BBBBBBB")
    user_input = request.form["user_input"]
    #call function from main.py here / send input over
    #time.sleep(10) #simulate processing time
    return render_template("output.html")

#@app.route('/output')
#def analysis_output():
    #return render_template('output.html')
 

if __name__ == "__main__":
    app.run(debug=True)