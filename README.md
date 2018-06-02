# Brutus

Brutus is a combination of Raspberry Pi 3 Model B, a fashion mannequin and AWS AI services. For now it can perform simple tasks such as detect objects in picture, facial recognition and respond to input in English. Further development is pending.

![Brutus at work](https://s3-eu-west-1.amazonaws.com/brutus-reko/github_picts/brutus-at-work.jpg "Brutus at work")

### Architecture and data flow process of Brutus
Brutus runs on Raspberry Pi 3 Model B. Additional computing power and libraries are used via Amazon Web Services AI services, such as Polly, Rekognition and Transcribe.
![Dataflow process and components used](https://s3-eu-west-1.amazonaws.com/brutus-reko/github_picts/dataflow.png "Dataflow with Brutus")

## Required components
For the setup to work you need the following components:
### 1. AWS account
All data processing is essentially done using AWS services. I recommend creating an IAM user with limited rights only to modules used. In my setup I created IAM user for Brutus with access to Polly, Rekognition, S3, Codecommit and Transcribe.
### 2. Raspberry Pi
Raspberry Pi is used to run Python scripts and storing
#### 2.1 Raspberry Pi Camera Module
Issue Brutus a command to do facial recognition. It takes picture using Camera module. You can also take picture with any other device if you wish to just use Rekognition or OpenCV for image processing without Raspberry Pi setup.
#### 2.2 Speaker connected to Raspberry Pi
Playback audio with speaker. Raspberry Pi 3 Model B supports Bluetooth speakers well, but I noticed that AUX cable is still the option that requires least effort.
#### 2.3 Powerbank for powering Raspberry Pi
I used a 6700 mAh Powerbank for powering Raspberry Pi. Depending on how long you want your setup to run, buy a bigger or smaller powerbank.
### 3. Auxiliary devices for connecting to Raspberry: screen, keyboard and mouse, or a laptop and RDP/VNC connection
Code is run on Raspberry Pi. You can either write code on another computer and transfer it to Raspberry via SSH or VNC connection, or connect directly to Raspberry Pi by plugging it into a screen and connecting a keyboard and mouse to it. Your choice.
### 4. Optional: Microphone for AWS Transcribe
In case you want to issue commands to Brutus by talking to it, you can record sound with microphone, save audio file as mp3, upload it to S3 bucket, and let Transcribe recognize words in the audio file. This part is optional, since getting a microphone to work with Raspberry Pi is not that easy, and the code is not described fully in code examples.

## Brutus' internal code structure
Brutus is composed of separate modules. AWS AI services are called in function calls using AWS Python SDK, authentication is done via aws-cli.
![Code structure used](https://s3-eu-west-1.amazonaws.com/brutus-reko/github_picts/codingstructure.png "Code structure")
Main program mainbrutus.py is used to call functions in separate modules, input is given either as hard-coded strings or via input from visual feed (PiCamera) or audio (microphone/mp3 file).

Information on [AWS CLI](https://aws.amazon.com/cli/?sc_channel=PS&sc_campaign=acquisition_ND&sc_publisher=google&sc_medium=command_line_b&sc_content=aws_cli_e&sc_detail=cli%20aws&sc_category=command_line&sc_segment=211498327597&sc_matchtype=e&sc_Country=ND&s_kwcid=AL!4422!3!211498327597!e!!g!!cli%20aws&ef_id=V31KegAAAOvVrniL:20180602125917:s) here.

Install with

    $ pip3 install awscli

## Usage
### Prerequisites
AWS Command Line Interface (aws-cli) needs to be setup with IAM user having rights to S3, Polly and Rekognition. Authentication can be done by either giving Access Key and Secret Access key as parameters in code, or by using awscli authentication. I chose aws-cli so I don't have to save access keys in code.

Python dependencies are to be installed, at least:
* boto3, botocore - for AWS Python SDK
* PiCamera - for Raspberry Pi camera
* numpy - for data processing
* Pillow, matplotlib - for image processing
* json, time, datetime - for data parsing
* requests - for downloading content
* vlc - for audio playback

I did not do full check on all required dependencies. This will later be fixed by dockerising the environment.

Install dependencies with pip:

    $ sudo pip3 install boto3 botocore picamera numpy Pillow matplotlib vlc

## Using a working program

    $ python3 mainbrutus.py

Program asks for input, you can try the following:
* "Brutus, introduce yourself"
* "Brutus, do facial recognition"
* "Brutus, why don't you have legs?"


## Further development
* Audio feedback has a weird lag. Has something to do with initialising vlc instance. Might be solved easily.
* Refactoring code: Many of the functions should be made into instances of classes. Some functions return two outputs now and do two or three things, which is not clean code.
* Response from facial recognition is not too elegant. Further if-statements need to be inserted, for example if no humans are detected in image, then do label detection. Otherwise do face detection.
* Only one person is now described in detect_faces response. Response from Polly returns data from all faces detected, but for now only labels for one person are parsed for output.
* I had too much trouble getting microphone to work in Raspberry Pi and getting Rasp to recognise microphone in startup as default audio input, but also defaulting audio output to a speaker. Also the time needed for Transcribe to process an audio file is 30-90 seconds, so there is a definite lag in response times if audio is used to input commands.
* Laser pointer needs to be inserted in Brutus' eye. Otherwise it doesn't look like a real cyborg.
* Implement AWS Lex so Brutus can act as a chatbot. Later combine this with Transcribe and Polly.

## Further resources on Internet
- [AWS Rekognition](https://aws.amazon.com/rekognition/)
- [AWS Polly](https://aws.amazon.com/polly/)
- [AWS Transcribe](aws.amazon.com/transcribe/)
- [Securing your Raspberry Pi](https://www.raspberrypi.org/documentation/configuration/security.md)
- [Connecting to Raspberry Pi with RealVNC Cloud](https://www.realvnc.com/en/raspberrypi/)
