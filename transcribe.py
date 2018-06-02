import boto3
import requests
from requests import get
from pprint import pprint
import json
import time

client=boto3.client('transcribe')

def start_transcribe(job_name, file):
    ''' Starts AWS Transcribe job with a given mp3 file in S3 bucket. '''
    response = client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode='en-US',
        MediaFormat='mp3',
        Media={
            'MediaFileUri': file
            }
    )

def get_job_status(job_name):
    ''' Gets job status from AWS Transcribe.'''
    response = client.get_transcription_job(
    TranscriptionJobName=job_name
    )
    return response['TranscriptionJob']['TranscriptionJobStatus']

def get_response(job_name, filetobedownloaded):
    ''' Gets file url from a JSON file with job results from AWS Transcribe.
    Downloads the file from Transcribe to local drive.'''
    response = client.get_transcription_job(
        TranscriptionJobName=job_name
    )
    dict = response['TranscriptionJob']
    url= dict['Transcript']['TranscriptFileUri']

    #Download file to local drive to be processed
    def download(url, filetobedownloaded):
        with open(filetobedownloaded, "wb") as file:
            response = get(url)
            file.write(response.content)
    download(url,filetobedownloaded)

def get_brutus_response(brutusresponse):
    ''' Processes def get_response given file, returns as text
    what was said in audio file that Transcribe processed.   '''
    with open(brutusresponse, 'r') as readfile:
        data = json.load(readfile)

    for item in data['results']['transcripts']:
        global brutus_input
        brutus_input = item['transcript']
        return 'Command given: ' + brutus_input

def transcribe_brutus(job,jobname_stat,brutusresponse):
    ''' Recursive function that prints out get_brutus_response
    for a job started when the job is completed.
    Used in conjuction with def start_transcribe.  '''
    if str(get_job_status(jobname_stat))=='COMPLETED':
        get_response(jobname_stat, brutusresponse)
        print(str(get_brutus_response(brutusresponse)))
    else:
        time.sleep(5)
        transcribe_brutus(jobname_stat)
