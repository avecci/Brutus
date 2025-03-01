# B.R.U.T.U.S. Backend

## Overview

B.R.U.T.U.S. is an AI robot that can do various functionalities such as image analysis text-to-speech conversion. This project includes a Raspberry Pi client, API backend interface created with FastAPI, and frontend built with ReactPY.

## Features

- **Image Analysis**: Detects labels, faces, and performs facial recognition using AWS Rekognition.
- **Text-to-Speech**: Converts text to speech using AWS Polly.


- **Raspberry Pi Client**: Terminal interface for commanding backend, e.g. take pictures or produce audio.
- **Backend**: FastAPI backend that wraps backend functionalities as API endpoints.
- **Frontend**: A ReactPY frontend using hooks to call FastAPI backend.

## Architecture
- uml
- api call flow
-openapi standard (localhost:8000/docs)


## Project Structure

```plaintext
.
├── backend
│   ├── api.py
│   ├── image_recognition.py
│   ├── logging_utils.py
│   ├── speech_generator.py
│   ├── tests
│   │   ├── test_image_analysis_unit.py
│   │   ├── test_image_analysis_integration.py
│   │   ├── test_api.py
│   │   ├── test_main.py
│   └── pyproject.toml
├── frontend
│   ├── main.py
│   └── static
│       └── styles
│           └── main.css
├── raspberrypi_client
│   └── brutus_pi.py
├── .pre-commit-config.yaml
└── README.md
```
## Installation
### Prerequisites
* Python 3.10 or higher
* AWS CLI configured with AWS SSO
* Poetry for dependency management

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/brutus.git
cd brutus
```
2. Install dependencies:
awscli
poetry (version >2.0.0)
poetry lock

raspberrypiclient:
sudo apt update && sudo apt upgrade
sudo apt-get install build-essential
sudo apt install libcap-dev libatlas-base-dev ffmpeg libopenjp2-7
sudo apt install libcamera-dev
sudo apt install libkms++-dev libfmt-dev libdrm-dev

3. Set up environment variables:
aws sso configure
export AWS_REGION=eu-central-1
export AWS_PROFILE=....

Run the backend:

Run the frontend:



## Usage
- aws credentials
- poetry instructions
poetry lock, poetry install
poetry run task test
poetry run task dev
## Monitoring
- logging in json, formatting
## Tests
- unit tests, integration tests, api tests
- poetry run tests
## Development
pre-commit run --all-files


# B.R.U.T.U.S.

## Introduction
## Features
- image recognition: labels, people, faces, master recognition
- speech generation and output via UI
- upcoming: arduino / servo motors, ruuvitag, bedrock/llm chatting via alexa dot
## Architecture
- uml
- api call flow
-openapi standard (localhost:8000/docs)
## Usage
- aws credentials
- poetry instructions
poetry lock, poetry install
poetry run task test
poetry run task dev
## Monitoring
- logging in json, formatting
## Tests
- unit tests, integration tests, api tests
- poetry run tests
## Development
pre-commit run --all-files
