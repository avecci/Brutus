"""Command Raspberry Pi and call Brutus backend API."""
import os
import sys
import tempfile
from time import sleep
from typing import Any, Dict, NoReturn

from PIL import Image

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # noqa: E402
import pygame  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from logging_utils import setup_logger  # noqa: E402

# Initialize logger
logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application configuration settings."""

    HOST: str = "0.0.0.0"
    PORT: int = 3000
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    APP_TITLE: str = "B.R.U.T.U.S."


settings = Settings()


class AudioHandler:
    """Handle audio playback operations."""

    def __init__(self) -> None:
        """Initialize pygame mixer once at startup."""
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    def play_audio(self, audio_path: str) -> None:
        """Play audio file from given path."""
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                sleep(0.1)
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Cleanup mixer resources."""
        try:
            pygame.mixer.quit()
        except Exception as e:
            logger.error(f"Error cleaning up mixer: {str(e)}")


class PictureHandler:
    """Take a picture and send it to backend via API call."""

    TARGET_IMAGE_PATH: str = "input/input_image.jpg"
    BACKEND_API_IMAGE_UPLOAD_URL: str = f"{settings.BACKEND_URL}/image/upload"
    COUNTDOWN_TIME: int = 5  # Wait time before capturing image

    def __init__(self) -> None:
        """Initialize the PictureHandler."""
        self.image_path: str = self.TARGET_IMAGE_PATH
        os.makedirs(os.path.dirname(self.TARGET_IMAGE_PATH), exist_ok=True)

    def capture_rotate_and_save(self, image_path: str) -> None:
        """Take a picture using rpicam-still and rotate it."""
        try:
            logger.info("Starting image capture with rpicam-still")

            # Create a temporary file for the original image
            temp_path = f"{image_path}.temp"

            # Countdown
            for i in range(self.COUNTDOWN_TIME, 0, -1):
                print(f"Countdown: {i}...")
                sleep(1)

            # Capture image with rpicam-still to temp file
            command = f"rpicam-still -n -o {temp_path}"
            result = os.system(command)

            if result != 0:
                raise RuntimeError(f"rpicam-still failed with exit code {result}")

            # Rotate the image using PIL
            with Image.open(temp_path) as img:
                rotated_img = img.rotate(
                    -90, expand=True
                )  # Rotate 90 degrees clockwise
                rotated_img.save(image_path)

            # Remove temporary file
            os.remove(temp_path)

            logger.debug(f"Image captured, rotated and saved to {image_path}")

        except Exception as e:
            logger.error(f"Error during image capture: {str(e)}")
            raise

    def upload_image(self, input_image: str) -> Dict[str, Any]:
        """Upload the captured image to backend."""
        logger.info(f"Attempting to upload image from {input_image}")
        try:
            with open(input_image, "rb") as f:
                response = requests.post(
                    self.BACKEND_API_IMAGE_UPLOAD_URL,
                    files={"file": (self.TARGET_IMAGE_PATH, f, "image/jpeg")},
                )
            response.raise_for_status()
            logger.info("Image upload successful")
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to upload image: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upload: {str(e)}")
            raise


class BackendHandler:
    """Handle backend API communication."""

    @classmethod
    def check_backend_health(cls) -> bool:
        """Verify backend is running and healthy."""
        try:
            response = requests.get(f"{settings.BACKEND_URL}/health")
            response.raise_for_status()
            health_data = response.json()
            return health_data.get("status") == "Healthy"
        except Exception as e:
            logger.error(f"Backend health check failed: {str(e)}")
            return False


class MenuHandler:
    """Handle terminal menu interface."""

    MENU_OPTIONS = {
        "1": "Take picture and upload",
        "2": "Introduce Brutus",
        "3": "Laugh",
        "4": "Generate speech",
        "q": "Quit",
    }

    def __init__(
        self, picture_handler: PictureHandler, audio_handler: AudioHandler
    ) -> None:
        """Initialize Menu."""
        self.picture_handler = picture_handler
        self.audio_handler = audio_handler

    def display_menu(self) -> None:
        """Display menu options."""
        print("\nBRUTUS TERMINAL INTERFACE")
        print("------------------------")
        for key, value in self.MENU_OPTIONS.items():
            print(f"{key}: {value}")

    def handle_option(self, choice: str) -> bool:
        """Handle menu selection. Returns False if should quit."""
        if choice == "1":
            try:
                self.picture_handler.capture_rotate_and_save(
                    PictureHandler.TARGET_IMAGE_PATH
                )
                self.picture_handler.upload_image(PictureHandler.TARGET_IMAGE_PATH)
                print("Picture captured and uploaded successfully!")
            except Exception as e:
                print(f"Error: {str(e)}")

        elif choice == "2":
            try:
                self.audio_handler.play_audio("audio/introduction.mp3")
            except Exception as e:
                print(f"Error playing introduction: {str(e)}")

        elif choice == "3":
            try:
                self.audio_handler.play_audio("audio/laugh.mp3")
            except Exception as e:
                print(f"Error playing introduction: {str(e)}")

        elif choice == "4":
            try:
                text = input("Enter text for Brutus to speak: ").strip()
                if text:
                    response = requests.post(
                        f"{settings.BACKEND_URL}/speech/generate",
                        params={"text": text},
                        stream=True,
                    )
                    response.raise_for_status()

                    # Save to temporary file and play
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    ) as temp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            temp_file.write(chunk)
                        temp_file.flush()

                        self.audio_handler.play_audio(temp_file.name)

                    # Cleanup
                    os.unlink(temp_file.name)

            except Exception as e:
                print(f"Error generating speech: {str(e)}")

        elif choice.lower() == "q":
            print("Goodbye!")
            return False

        return True

    def run(self) -> NoReturn:
        """Run the menu loop."""
        while True:
            self.display_menu()
            choice = input("\nEnter your choice: ").strip()

            if choice not in self.MENU_OPTIONS:
                print("Invalid option, please try again.")
                continue

            if not self.handle_option(choice):
                break


if __name__ == "__main__":
    audio = None
    try:
        logger.info("Initializing Brutus")

        # Check backend health first
        if not BackendHandler.check_backend_health():
            logger.error("Backend is not healthy, aborting")
            raise SystemExit("Backend not available")

        # Initialize handlers
        audio = AudioHandler()
        picture_handler = PictureHandler()

        # Play sound to tell user Brutus is ready
        audio.play_audio("audio/by_your_command.mp3")
        sleep(1)

        # Start menu interface
        menu = MenuHandler(picture_handler, audio)
        menu.run()

        sys.exit(0)

    except SystemExit as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        sys.exit(1)
    finally:
        if audio:
            audio.cleanup()
