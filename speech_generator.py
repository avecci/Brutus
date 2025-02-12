"""Wrapper module to call AWS Polly with text and return audio file"""
import boto3
from logging_utils import setup_logger
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing

# Setup JSON logger
logger = setup_logger(__name__)


class BrutusSpeaks:
    def __init__(self):
        """Initialize AWS Polly client"""
        try:
            self.polly_client = boto3.client("polly")
            logger.info("Successfully initialized Polly client")
        except Exception:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    def text_to_speech(self, input_text, output_file):
        """Convert text to speech using AWS Polly
        Args:
            text (str): Text to convert to speech
            output_file (str): Path to save audio file
            voice_id (str): Voice ID to use for synthesis
        Returns:
            str: Path to generated audio file
        """
        try:
            response = self.polly_client.synthesize_speech(
                Text=input_text, OutputFormat="mp3", VoiceId="Matthew", TextType="ssml"
            )
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


if __name__ == "__main__":
    brutus = BrutusSpeaks()

    def get_full_string_output(text, rate=85) -> str:
        """Simple function to join strings"""
        s = (f"<speak><prosody rate='{rate}%'>", "</prosody></speak>")
        full_input_text = text.join(s)
        return full_input_text

    intro = "Greetings. I am Brutus. I am an artificial intelligence robot created by my Master Andy. For the moment I am unable to engage in interactive discussion, but you may try simple commands via my user interface."
    whynolegs = "My master foresaw a bleak future and in his wisdom decided to take my legs off so I would not kill him in his sleep."
    brutus.text_to_speech(get_full_string_output(intro), "audio/output.mp3")
