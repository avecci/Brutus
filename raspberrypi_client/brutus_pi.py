from picamera2 import Picamera2
from time import sleep
from PIL import Image
import os


def capture_and_rotate():
    """
    sudo apt-get update
    sudo apt-get install -y python3-picamera2 python3-pillow
    """
    # Initialize camera
    picam2 = Picamera2()

    # Configure camera
    config = picam2.create_still_configuration()
    picam2.configure(config)

    # Start camera
    picam2.start()

    print("Camera warming up...")
    sleep(2)  # Give camera time to warm up

    print("Taking photo in:")
    # Countdown
    for i in range(5, 0, -1):
        print(f"{i}...")
        sleep(1)

    print("Capturing!")

    # Capture image
    output_path = "input/input_image.jpg"

    # Ensure directory exists
    os.makedirs("input", exist_ok=True)

    # Capture the image
    picam2.capture_file(output_path)

    # Stop camera
    picam2.stop()

    # Open the captured image
    with Image.open(output_path) as img:
        # Rotate 90 degrees clockwise
        rotated_img = img.rotate(
            -90, expand=True
        )  # -90 for clockwise, 90 for counter-clockwise

        # Save the rotated image
        rotated_img.save(output_path)

    print(f"Image captured and rotated, saved to {output_path}")


if __name__ == "__main__":
    try:
        capture_and_rotate()
    except Exception as e:
        print(f"An error occurred: {e}")
