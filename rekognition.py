import boto3
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image #pip install Pillow
import numpy as np

client=boto3.client('rekognition','eu-west-1')

def detectlabels(s3bucket,kuvatiedosto,textoutputfile):
    ''' Fetches response from AWS Rekognition about detected labels
    in a picture and writes the output to an outputfile.  '''
    response = client.detect_labels(Image={'S3Object':{'Bucket':s3bucket,'Name':kuvatiedosto}})
    outputfile = open(textoutputfile, 'w')
    for label in response['Labels']:
        outputfile.write(label['Name']+ ": " + str(label['Confidence']))
        outputfile.write("\n")
    outputfile.close()
    for label in response['Labels']:
        print(label['Name']+ ": " + str(label['Confidence']))

def show_original_picture(kuvatiedosto):
    ''' Simple function to open a picture from local drive.
    Uses Pillow library, might require installing Imagemagick.'''
    img = Image.open(kuvatiedosto)
    img.show()

def compare_faces(s3bucket, sourceimage, targetimage):
    ''' Compares faces in two images, returns confidence level of the two having the same face in the image.'''
    response = client.compare_faces(
        SourceImage={
            'S3Object': {
                'Bucket': s3bucket,
                'Name': sourceimage
            }
        },
        TargetImage={
            'S3Object': {
                'Bucket': s3bucket,
                'Name': targetimage
            }
        },
        SimilarityThreshold=0.7
    )
    for detail in response['FaceMatches']:
        if detail['Similarity'] is not None:
            return float(detail['Similarity'])
        else:
            return 0

def detectfaces(s3bucket,kuvatiedosto,kuvatiedosto2,textoutputfile2):
    ''' Detects faces in image, returns details of face as outputfile and
    draws a box around the first image detected in the image. Writes details in the image also. '''
    global listofdetails
    global imwidth
    global imheight
    global imleft
    global imtop

    response = client.detect_faces(Image={'S3Object':{'Bucket':s3bucket,'Name':kuvatiedosto}},Attributes=['ALL'])
    outputfile2 = open(textoutputfile2, 'w')
    for faceDetail in response['FaceDetails']:
        outputfile2.write(json.dumps(faceDetail, indent=4, sort_keys=True))
    outputfile2.close()

    listofdetails = []
    for faceDetail in response['FaceDetails']:
        listofdetails.append('Approximate age: ')
        listofdetails.append('    ' + str(faceDetail['AgeRange']['Low']) + '-' + str(faceDetail['AgeRange']['High']) + ' years')
        listofdetails.append('Beard: ')
        listofdetails.append('    ' + str(faceDetail['Beard']))
        listofdetails.append('Mustache: ')
        listofdetails.append('    ' + str(faceDetail['Mustache']))
        listofdetails.append('Emotions: ')
        for i in range(0,len(faceDetail['Emotions'])):
            listofdetails.append('    ' + str(faceDetail['Emotions'][i]))
        listofdetails.append('Smile: ')
        listofdetails.append('    ' + str((faceDetail['Smile'])))
        listofdetails.append('Eyeglasses: ')
        listofdetails.append('    ' + str(faceDetail['Eyeglasses']))
        listofdetails.append('Sunglasses: ')
        listofdetails.append('    ' + str(faceDetail['Sunglasses']))
        listofdetails.append('EyesOpen: ')
        listofdetails.append('    ' + str(faceDetail['EyesOpen']))
        listofdetails.append('Gender: ')
        listofdetails.append('    ' + str(faceDetail['Gender']))

    # Get bounding box parameters for image manipulation
        imwidth  = faceDetail['BoundingBox']['Width']
        imheight = faceDetail['BoundingBox']['Height']
        imleft   = faceDetail['BoundingBox']['Left']
        imtop    = faceDetail['BoundingBox']['Top']

    # Open picture and display box on detected face:
    with Image.open(kuvatiedosto) as img:
        width,height = img.size

        img = np.array(Image.open(kuvatiedosto), dtype=np.uint8)
    # Create figure and axes
        fig,ax = plt.subplots(1)
    # Display the image
        ax.imshow(img)

    # Create a Rectangle patch
        rect = patches.Rectangle((imleft*width,imtop*height),imwidth*width,imheight*height,linewidth=1,edgecolor='b',facecolor='none')

    # Add the patch to the Axes and insert text from Rekognition
        ax.add_patch(rect)
        dpi=300
        for j in range(0,len(listofdetails)):
            plt.text((width+50),height/4+j*75,listofdetails[j],fontsize=7)
            plt.savefig(kuvatiedosto2, orientation='portrait', figsize=(width/dpi,height/dpi), dpi = dpi)
    plt.show()
