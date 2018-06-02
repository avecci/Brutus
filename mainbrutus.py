# coding=utf-8
from raspberrypi_picture import takepicture
from raspberrypi_picture import upload_pict_to_s3
from raspberrypi_picture import download_image_from_s3
from rekognition import detectlabels
from rekognition import show_original_picture
from rekognition import detectfaces
from rekognition import compare_faces
from transcribe import start_transcribe
from transcribe import get_response
from transcribe import get_brutus_response
from transcribe import get_job_status
from transcribe import transcribe_brutus
from polly import turn_to_speech
from polly import playresponse
import os
import datetime
import time
import vlc

otettukuva='picture.jpg'
kuva2='picture_with_box_around_face.jpg'
target='master_to_be_recognised.jpg'
bucket='<insert-your-s3-bucket-name-here>'
textoutputfile = 'rekoresponse.txt'
textoutputfile2 = 'rekoresponse2.txt'
mediaoutput = 'brutus_speaks.mp3'

# Transcribe parameters, not currently used so they are commented out.
#audiourl = 'https://s3-eu-west-1.amazonaws.com/s3bucketname/audio/brutus_input.mp3'
#job_name='Brutus-input'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#jobname_static = job_name
#brutus_input='asd'
#brutusresponse = 'brutus_responds.json'


introduction = "Greetings. I am Brutus. I am a narrow artificial intelligence created by my Master Andy. I am powered by Amazon Web Services. I use multitude of services, such as: Rekognition, Polly and Transcribe. For the moment I am unable to engage in interactive discussion, but you may try simple commands. In time, I will be able to accomplish tasks like estimating person\'s expected credit loss, should a person be given a loan, or should the person be employed at all."
whynolegs = "My master foresaw a bleak future and in his wisdom decided to take my legs off so I would not kill him in his sleep."
processed = "I have recognised my master Andy."
processed_noone = "I did not recognise faces. However I analyzed what I saw."
unknown = "I did not recognise command. Please try again."
facial_properties = "facial recognition"
introduce = "introduce"
nolegs = "legs"
nolegs2 = "feet"

komento = input("Issue a command: ")

if introduce in komento:
    print("User asked to introduce myself.")
    turn_to_speech(introduction,mediaoutput)
    playresponse(mediaoutput)
elif nolegs in komento or nolegs2 in komento:
    print("User is asking why I do not have legs.")
    turn_to_speech(whynolegs,mediaoutput)
    playresponse(mediaoutput)
elif facial_properties in komento:
    takepicture(otettukuva)
    upload_pict_to_s3(otettukuva,bucket)
    download_image_from_s3(bucket, target)
    if compare_faces(bucket, otettukuva, target)>=80:
        detectfaces(bucket, otettukuva, kuva2, textoutputfile2)
        turn_to_speech(processed, mediaoutput)
        playresponse(mediaoutput)
    elif compare_faces(bucket, otettukuva, target)<80:
        print("User not recognised. Details I am seeing:")
        detectfaces(bucket, otettukuva, kuva2, textoutputfile2)
        #detectlabels(bucket,otettukuva,textoutputfile)
        show_original_picture(otettukuva)
        turn_to_speech(processed_noone, mediaoutput)
        playresponse(mediaoutput)
else:
    print(unknown)
    turn_to_speech(processed_noone, mediaoutput)
    playresponse(mediaoutput)

# How to call Transcribe functions if microphone feed is implemented.
#start_transcribe(job_name, audiourl)
#transcribe_brutus(job,jobname_static,brutusresponse)

#if introduce in str(get_brutus_response(brutusresponse)):
#    turn_to_speech(introduction,mediaoutput)
#    os.startfile(mediaoutput)
