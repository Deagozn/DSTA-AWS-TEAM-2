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
    return render_template('scamtextdetector.html')

@app.route('/scamcalldet')
def ScamCallDetect():
    return render_template('scamcalldetector.html')

@app.route('/output', methods=["POST"])
def ScamTextInput():
    time.sleep(1) #simulate long processing time
    user_input = request.form["user_input"]
    print(user_input)
    #call function from main.py here / send input over
    #assume output as dict
    sample_output = {'text':'We are the licensed money lender in Singapore. is amongst the pioneer loan service providers in Singapore. Are you have financial problems? Contact us for more informations and our friendly staff will be glad to assist you with your financial needs.',
                     'status':'yes',
                     'confidence':'99.6'}
    return render_template("output.html",text=sample_output['text'],status=sample_output['status'],confidence=sample_output['confidence'])


if __name__ == "__main__":
    app.run(debug=True)