"""Wrapper module to call AWS Polly with text and return audio file."""
import os
from contextlib import closing
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from logging_utils import setup_logger

# Setup JSON logger
logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()


class BrutusSpeechGenerator:
    """A class to handle text-to-speech conversion using AWS Polly.

    Provides functionality to convert text to speech using AWS Polly service.
    It handles the initialization of the AWS Polly client and provides methods to
    synthesize speech from text input.

    Example:
        >>> generator = BrutusSpeechGenerator()
        >>> output_path = generator.text_to_speech(
        ...     "Hello, world!",
        ...     "output/hello.mp3"
        ... )
    """

    def __init__(self) -> None:
        """Initialize AWS Polly client or die trying."""
        try:
            profile_name = os.getenv("AWS_PROFILE")
            region_name = os.getenv("AWS_REGION", "eu-central-1")
            if not profile_name:
                logger.error("AWS_PROFILE not found in .env file")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            self.polly_client = session.client("polly")
            logger.info("Successfully initialized Polly client")
        except Exception:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    @staticmethod
    def get_full_string_output(text: str, speech_rate: int) -> str:
        """Add speech rate to SSML text.

        Args:
            text: The text to be wrapped in SSML tags
            speech_rate: The rate of speech (percentage)

        Returns:
            String with SSML tags and speech rate applied
        """
        s = (f"<speak><prosody rate='{speech_rate}%'>", "</prosody></speak>")
        full_input_text = text.join(s)
        return full_input_text

    def text_to_speech(
        self, input_text: str, output_file: str, speech_rate: int = 85
    ) -> Optional[str]:
        """Convert text to speech using AWS Polly.

        This method takes input text and converts it to speech using Matthew voice,
        and saves the resulting audio as an MP3 file.

        Args:
            text (str): Text to convert to speech
            output_file (str): Path to save audio file
            voice_id (str): Voice ID to use for synthesis
        Returns:
            str: Path to generated audio file
        """
        try:
            input_text = self.get_full_string_output(input_text, speech_rate)
            response = self.polly_client.synthesize_speech(
                Text=input_text, OutputFormat="mp3", VoiceId="Matthew", TextType="ssml"
            )
            logger.info("Successfully initialized Polly client")
        except (BotoCoreError, ClientError):
            logger.error("Unable to synthesize speech", exc_info=True)
            return None

        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                output = output_file
                try:
                    with open(output, "wb") as file:
                        file.write(stream.read())
                        logger.info(
                            "Audio file saved successfully",
                            extra={"output_file": output},
                        )
                except OSError:
                    logger.error("Unable to save audio file", exc_info=True)
                    return None

            return output_file
        return None
