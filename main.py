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

IbucketName = "bucktest5354-" + awsRegion

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

    display(detectLabelsResponse)

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
        detected_entities_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_entities['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print()
        print(Ctext)
        print()
        display(detected_entities_df)

    elif identifier.lower() == "key phrases":
        detected_key_phrases = comprehend.detect_key_phrases(Text = Ctext, LanguageCode = 'en')
        detected_key_phrases_df = pd.DataFrame([ [entity['Text'], entity['Score']] for entity in detected_key_phrases['KeyPhrases']],
                columns=['Text', 'Score'])
        print("This was the text analyzed:")
        print()
        print(Ctext)
        print()
        display(detected_key_phrases_df)
    
    elif identifier.lower() == "dominant language":
        detected_language = comprehend.detect_dominant_language(Text = Ctext)
        detected_language_df = pd.DataFrame([ [code['LanguageCode'], code['Score']] for code in detected_language['Languages']],
                columns=['Language Code', 'Score'])
        print("This was the text analyzed:")
        print()
        print(Ctext)
        print()
        display(detected_language_df)

    elif identifier.lower() == "emotional sentiment":
        detected_sentiment = comprehend.detect_sentiment(Text = Ctext, LanguageCode = 'en')
        predominant_sentiment = detected_sentiment['Sentiment']
        detected_sentiments_df = pd.DataFrame([ [sentiment, detected_sentiment['SentimentScore'][sentiment]] for sentiment in detected_sentiment['SentimentScore']],
                columns=['Language Code', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        print("The predominant sentiment is {}.".format(predominant_sentiment))
        print()
        display(detected_sentiments_df)

    elif identifier.lower() == "syntax":
        detected_syntax = comprehend.detect_syntax(Text = Ctext, LanguageCode = 'en')
        detected_syntax_df = pd.DataFrame([ [part['Text'], part['PartOfSpeech']['Tag'], part['PartOfSpeech']['Score']] for part in detected_syntax['SyntaxTokens']],
                columns=['Text', 'Part Of Speech', 'Score'])
        print("This was the text analyzed:")
        print()
        print(Ctext)
        print()
        display(detected_syntax_df)
    
    elif identifier.lower() == "pii":
        detected_pii_entities = comprehend.detect_pii_entities(Text = Ctext, LanguageCode = 'en')
        detected_pii_entities_df = pd.DataFrame([ [entity['Type'], entity['Score']] for entity in detected_pii_entities['Entities']],
                columns=['Type', 'Score'])
        print("This was the text analyzed:")
        print()
        print(Ctext)
        print()
        display(detected_pii_entities_df)
    
    elif identifier.lower() == "pii labels":
        detected_pii_labels = comprehend.contains_pii_entities(Text = Ctext, LanguageCode = 'en')
        detected_pii_labels_df = pd.DataFrame([ [entity['Name'], entity['Score']] for entity in detected_pii_labels['Labels']],
                columns=['Name', 'Score'])
        print("This was the text analyzed:")
        print()
        print(Ctext)
        print()
        display(detected_pii_labels_df)
