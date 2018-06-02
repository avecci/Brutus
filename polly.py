import boto3
import botocore
from contextlib import closing
import os
import vlc
import time

client = boto3.client('polly')

def turn_to_speech(input, outputfile):
    ''' Fetches audiostream from AWS Polly, writes it as mp3 file.'''
    response = client.synthesize_speech(
        OutputFormat='mp3',
        Text=input,
        TextType='text',
        VoiceId='Matthew'
        )

    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:
            output = outputfile

            try:
                # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                    # Could not write to file, exit gracefully
                    print(error)
                    sys.exit(-1)
    print(input)

def playresponse(outputfile):
    ''' Opens VLC instance and plays mp3 file for 30 seconds. '''
    instance = vlc.Instance()
#Create a MediaPlayer with the default instance
    player = instance.media_player_new()
#Load the media file
    media = instance.media_new(outputfile)
#Add the media to the player
    player.set_media(media)
    player.play()
    time.sleep(30)
