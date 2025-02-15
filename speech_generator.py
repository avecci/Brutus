"""Wrapper module to call AWS Polly with text and return audio file"""
import os
import boto3
from logging_utils import setup_logger
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing

# Setup JSON logger
logger = setup_logger(__name__)


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

    def __init__(self):
        """Initialize AWS Polly client"""
        try:
            self.polly_client = boto3.client("polly")
            logger.info("Successfully initialized Polly client")
        except Exception:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    @staticmethod
    def get_full_string_output(text, speech_rate) -> str:
        """Add speech rate"""
        s = (f"<speak><prosody rate='{speech_rate}%'>", "</prosody></speak>")
        full_input_text = text.join(s)
        return full_input_text

    def text_to_speech(self, input_text, output_file, speech_rate=85):
        """Convert text to speech using AWS Polly
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
