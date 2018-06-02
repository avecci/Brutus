import picamera
import boto3
import botocore

def takepicture(pict_name):
    ''' Take a single picture with Raspberry Pi camera.
    Requires camera to be set on in raspi-config. '''
    camera = picamera.PiCamera()
    camera.capture(pict_name)

def upload_pict_to_s3(pict_name, bucket_name):
    ''' Uploads a file to AWS S3 bucket.'''
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(pict_name, pict_name)

def download_image_from_s3(bucket, file):
    ''' Downloads a file from AWS S3 bucket.'''    
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket).download_file(file, file)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
