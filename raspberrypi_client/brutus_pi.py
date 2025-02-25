"""Command Raspberry Pi and call Brutus backend API."""
from time import sleep

import requests
from picamera2 import Picamera2
from PIL import Image

from logging_utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)
output_path = "input/input_image.jpg"


def capture_and_rotate(output_path: str):
    """Ensure these libraries are included in system installation.

    sudo apt-get update
    sudo apt-get install -y python3-picamera2 python3-pillow

    Take a picture and rotate it 90 degrees as camera is rotated.
    """
    # Initialize camera
    logger.info("Initializing camera")
    picam2 = Picamera2()

    # Configure camera
    config = picam2.create_still_configuration()
    picam2.configure(config)

    # Start camera
    picam2.start()

    logger.info("Camera warming up...")
    sleep(2)  # Give camera time to warm up

    logger.info("Starting countdown for photo capture")
    # Countdown
    for i in range(5, 0, -1):
        logger.debug(f"Countdown: {i}...")
        print(f"{i}...")  # Keep print for user feedback
        sleep(1)

    logger.info("Capturing photo")
    print("Capturing!")  # Keep print for user feedback

    try:
        # Capture the image
        picam2.capture_file(output_path)
        logger.debug(f"Image captured and saved to {output_path}")

        # Stop camera
        picam2.stop()
        logger.debug("Camera stopped")

        # Open the captured image
        with Image.open(output_path) as img:
            # Rotate 90 degrees clockwise
            rotated_img = img.rotate(
                -90, expand=True
            )  # -90 for clockwise, 90 for counter-clockwise

            # Save the rotated image
            rotated_img.save(output_path)
            logger.info(f"Image rotated and saved to {output_path}")

        return {f"Image captured and rotated, saved to {output_path}"}

    except Exception as e:
        logger.error(f"Error during image capture and rotation: {str(e)}")
        raise


def upload_image(input_image):
    """Upload the captured image to backend."""
    logger.info(f"Attempting to upload image from {input_image}")
    try:
        with open(input_image, "rb") as f:
            response = requests.post(
                "http://localhost:8000/upload/image",
                files={"file": ("image.jpg", f, "image/jpeg")},
            )
        logger.info("Image upload successful")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to upload image: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting image capture and upload process")
        capture_and_rotate(output_path)
        upload_image(output_path)
        logger.info("Process completed successfully")

    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
