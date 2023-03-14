import boto3
import time
import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display, Image as IImage
import sagemaker
from sagemaker import get_execution_role
import pandas as pd
import numpy as np

mySession = boto3.session.Session()
awsRegion = mySession.region_name
role = get_execution_role()

rekognition = boto3.client('rekognition')
comprehend = boto3.client(service_name='comprehend', region_name = awsRegion)
s3 = boto3.client('s3')

tempFolder = 'm1tmp/'

IbucketName = "" + awsRegion

def drawBoundingBoxes (sourceImage, boxes):
    # blue, green, red, grey
    colors = ((255,255,255),(255,255,255),(76,182,252),(52,194,123))
    
    # Download image locally
    imageLocation = tempFolder+os.path.basename(sourceImage)
    s3.download_file(IbucketName, sourceImage, imageLocation)

    # Draws BB on Image
    bbImage = Image.open(imageLocation)
    draw = ImageDraw.Draw(bbImage)
    width, height = bbImage.size
    col = 0
    maxcol = len(colors)
    line= 3
    for box in boxes:
        x1 = int(box[1]['Left'] * width)
        y1 = int(box[1]['Top'] * height)
        x2 = int(box[1]['Left'] * width + box[1]['Width'] * width)
        y2 = int(box[1]['Top'] * height + box[1]['Height']  * height)
        
        draw.text((x1,y1),box[0],colors[col])
        for l in range(line):
            draw.rectangle((x1-l,y1-l,x2+l,y2+l),outline=colors[col])
        col = (col+1)%maxcol
    
    imageFormat = "PNG"
    ext = sourceImage.lower()
    if(ext.endswith('jpg') or ext.endswith('jpeg')):
        imageFormat = 'JPEG'

    bbImage.save(imageLocation,format=imageFormat)

    display(bbImage)


def rekog(Dimage,detect): # image to be in STR format ; detect to be in LIST format

    display(IImage(url=s3.generate_presigned_url('get_object', Params={'Bucket': IbucketName, 'Key': Dimage}))) #dispay the image

    detectLabelsResponse = rekognition.detect_labels(
    Image=
    {
        'S3Object': 
        {
            'Bucket': IbucketName,
            'Name': Dimage,
        }
    }
    )
    for label in detectLabelsResponse["Labels"]:
        if(label["Name"] in detect):
            print("Detected object:")
            print("- {} (Confidence: {})".format(label["Name"], label["Confidence"]))
    
    boxes = []
    objects = detectLabelsResponse['Labels']
    for obj in objects:
        for einstance in obj['Instances']:
            boxes.append((obj['Name'], einstance['BoundingBox']))

    drawBoundingBoxes(Dimage, boxes)

def compre(Ctext, identifier):  # Ctext is the text to be analyzed ; identifier is the tyoe of data to be analyzed
    if identifier.lower() == "named entities":
        detected_entities = comprehend.detect_entities(Text = Ctext, LanguageCode = 'en')
        detectec_entities_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_entities['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detectec_entities_df)

    elif identifier.lower() == "key phrases":
        detected_key_phrases = comprehend.detect_key_phrases(Text = Ctext, LanguageCode = 'en')
        detectec_key_phrases_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_key_phrases['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detectec_key_phrases_df)
    
    elif identifier.lower() == "dominant language":
        detected_language = comprehend.detect_dominant_language(Text = Ctext, LanguageCode = 'en')
        detectec_language_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_language['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detectec_language_df)

    elif identifier.lower() == "emotional sentiment":
        detected_sentiment = comprehend.detect_sentiment(Text = Ctext, LanguageCode = 'en')
        predominant_sentiment = detected_sentiment['Sentiment']
        detectec_sentiment_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_sentiment['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        print("The predominant sentiment is {}.".format(predominant_sentiment))
        print()
        display(detectec_sentiment_df)

    elif identifier.lower() == "syntax":
        detected_syntax = comprehend.detect_syntax(Text = Ctext, LanguageCode = 'en')
        detectec_syntax_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_syntax['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detectec_syntax_df)
    
    elif identifier.lower() == "pii":
        detected_pii_entities = comprehend.detect_pii_entities(Text = Ctext, LanguageCode = 'en')
        detectec_pii_entities_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_pii_entities['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detectec_pii_entities_df)
    
    elif identifier.lower() == "pii labels":
        detected_pii_labels = comprehend.contains_pii_entities(Text = Ctext, LanguageCode = 'en')
        detectec_pii_labels_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_pii_labels['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detectec_pii_labels_df)

