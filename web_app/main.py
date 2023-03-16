import boto3
import time
import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display, Image as IImage
import sagemaker
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
import json
import tarfile



mySession = boto3.session.Session()
awsRegion = mySession.region_name
role = get_execution_role()

#rekognition = boto3.client('rekognition')
comprehend = boto3.client(service_name='comprehend', region_name = awsRegion)
transcribe = boto3.client('transcribe')
s3c = boto3.client('s3')
s3r = boto3.resource('s3')

#tempFolder = 'm1tmp/' #not working atm

#IbucketName = "bucket9069-" + awsRegion

# def drawBoundingBoxes (sourceImage, boxes):
#     # blue, green, red, grey
#     colors = ((255,255,255),(255,255,255),(76,182,252),(52,194,123))
    
#     # Download image locally
#     imageLocation = tempFolder+os.path.basename(sourceImage)
#     s3c.download_file(IbucketName, sourceImage, imageLocation)

#     # Draws BB on Image
#     bbImage = Image.open(imageLocation)
#     draw = ImageDraw.Draw(bbImage)
#     width, height = bbImage.size
#     col = 0
#     maxcol = len(colors)
#     line= 3
#     for box in boxes:
#         x1 = int(box[1]['Left'] * width)
#         y1 = int(box[1]['Top'] * height)
#         x2 = int(box[1]['Left'] * width + box[1]['Width'] * width)
#         y2 = int(box[1]['Top'] * height + box[1]['Height']  * height)
        
#         draw.text((x1,y1),box[0],colors[col])
#         for l in range(line):
#             draw.rectangle((x1-l,y1-l,x2+l,y2+l),outline=colors[col])
#         col = (col+1)%maxcol
    
#     imageFormat = "PNG"
#     ext = sourceImage.lower()
#     if(ext.endswith('jpg') or ext.endswith('jpeg')):
#         imageFormat = 'JPEG'

#     bbImage.save(imageLocation,format=imageFormat)

#     display(bbImage)


# def rekog(Dimage,detect): # image to be in STR format ; detect to be in LIST format

#     display(IImage(url=s3c.generate_presigned_url('get_object', Params={'Bucket': IbucketName, 'Key': Dimage}))) #dispay the image

#     detectLabelsResponse = rekognition.detect_labels(
#     Image=
#     {
#         'S3Object': 
#         {
#             'Bucket': IbucketName,
#             'Name': Dimage,
#         }
#     }
#     )

#     display(detectLabelsResponse)

#     for label in detectLabelsResponse["Labels"]:
#         if(label["Name"] in detect):
#             print("Detected object:")
#             print("- {} (Confidence: {})".format(label["Name"], label["Confidence"]))
    
#     boxes = []
#     objects = detectLabelsResponse['Labels']
#     for obj in objects:
#         for einstance in obj['Instances']:
#             boxes.append((obj['Name'], einstance['BoundingBox']))

#     drawBoundingBoxes(Dimage, boxes)

def compre(Ctext, identifier):  # Ctext is the text to be analyzed ; identifier is the tyoe of data to be analyzed
    if identifier.lower() == "named entities":
        detected_entities = comprehend.detect_entities(Text = Ctext, LanguageCode = 'en')
        detected_entities_df = pd.DataFrame([ [entity['Text'], entity['Type'], entity['Score']] for entity in detected_entities['Entities']],
                columns=['Text', 'Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detected_entities_df)

    elif identifier.lower() == "key phrases":
        detected_key_phrases = comprehend.detect_key_phrases(Text = Ctext, LanguageCode = 'en')
        detected_key_phrases_df = pd.DataFrame([ [entity['Text'], entity['Score']] for entity in detected_key_phrases['KeyPhrases']],
                columns=['Text', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detected_key_phrases_df)
    
    elif identifier.lower() == "dominant language":
        detected_language = comprehend.detect_dominant_language(Text = Ctext)
        detected_language_df = pd.DataFrame([ [code['LanguageCode'], code['Score']] for code in detected_language['Languages']],
                columns=['Language Code', 'Score'])
        print("This was the text analyzed:")
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
        print(Ctext)
        print()
        display(detected_syntax_df)
    
    elif identifier.lower() == "pii":
        detected_pii_entities = comprehend.detect_pii_entities(Text = Ctext, LanguageCode = 'en')
        detected_pii_entities_df = pd.DataFrame([ [entity['Type'], entity['Score']] for entity in detected_pii_entities['Entities']],
                columns=['Type', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detected_pii_entities_df)
    
    elif identifier.lower() == "pii labels":
        detected_pii_labels = comprehend.contains_pii_entities(Text = Ctext, LanguageCode = 'en')
        detected_pii_labels_df = pd.DataFrame([ [entity['Name'], entity['Score']] for entity in detected_pii_labels['Labels']],
                columns=['Name', 'Score'])
        print("This was the text analyzed:")
        print(Ctext)
        print()
        display(detected_pii_labels_df)

def trscbe(job_name, file, m_format): #all variables should be in STR form
    transcribe.start_transcription_job(
        TranscriptionJobName = job_name,
        Media = {'MediaFileUri': file},
        MediaFormat = m_format,
        LanguageCode = 'en-US',
        OutputBucketName = "bucket9069"
    )
    max_tries = 60
    while max_tries > 0:
        max_tries -= 1
        job = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']:
            print(f"Job {job_name} is {job_status}.")
            if job_status == 'COMPLETED':
                print()
            break
        else:
            print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)
    
    content_object = s3r.Object('bucket9069', job_name + '.json')
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    transcript_df = pd.DataFrame([ [entity['transcript']] for entity in json_content['results']['transcripts']],
                columns=[''])
    print("Transcribed: ")
    print(transcript_df.to_string(index=False))

def tar_decomp(c_file_name,e_file_name,location):      # c_file_name is the file to be decompressed, e_file_name is the file to be extracted, location is where to store the extracted file
    file = tarfile.open(c_file_name)
    file.extract(e_file_name,location)
    file.close()

def scam_detector(job_name, file):
    start_response = comprehend.start_document_classification_job(
        JobName = job_name,
        DocumentClassifierArn = 'arn:aws:comprehend:us-east-1:249986139069:document-classifier/scamdetect/version/v1',
        InputDataConfig = 
        {
            'S3Uri':file,
            'InputFormat':'ONE_DOC_PER_FILE',
        },
        OutputDataConfig = 
        {
            'S3Uri':'s3://dcyberv6'
        },
        DataAccessRoleArn='arn:aws:iam::249986139069:role/service-role/AmazonComprehendServiceRoleS3FullAccess-90695'
    )
    return start_response

def create_text_file(text_for_file,file_name):
    f = open("C:\code stuff\Code Stuff\dsta code\load_text\{}.txt".format(file_name),"w")
    f.write(text_for_file)
    f.close()

def file_upload(file_name, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3c.upload_file(file_name, "textupload9069", object_name)





def main(input_str):             #Wrtie main code here
    print("start")
    file_n = "file1"
    create_text_file(input_str, file_n)
    time.sleep(0.5)
    file_upload("C:\code stuff\Code Stuff\dsta code\load_text\{}.txt".format(file_n))
    time.sleep(0.5)
    scam_detector("job1", "s3://textupload9069/{}.txt".format(file_n))
    time.sleep(380)
    # s3c.download_file("dcyberv3", "output.tar.gz", "D:\Code Stuff\python\DSTA AWS Code\output.tar.gz")
    # tar_decomp("D:\Code Stuff\python\DSTA AWS Code\output.tar.gz","predictions.jsonl","C:\code stuff\Code Stuff\dsta code\output")
    # result = json.loads("C:\code stuff\Code Stuff\dsta code\output\predictions.jsonl")
    # print("end")
    # os.remove("C:\code stuff\Code Stuff\dsta code\output\predictions.jsonl") #uncomment this if eveth else works
    # return result["Classes"][0]
    


if __name__ == "__main__":
    main("")
