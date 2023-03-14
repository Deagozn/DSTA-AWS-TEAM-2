import boto3
import time
import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display, Image as IImage

mySession = boto3.session.Session()
awsRegion = mySession.region_name

rekognition = boto3.client('rekognition')
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


