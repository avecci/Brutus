import logging
import uuid
from pathlib import Path
from image_recognition import ImageAnalyzer

# Create a filter to add request_id if not present
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


# Get the logger
logger = logging.getLogger(__name__)


def main():
    request_id = str(uuid.uuid4())
    extra = {"request_id": request_id}

    try:
        logger.info("Initializing image analyzer", extra=extra)
        analyzer = ImageAnalyzer()

        # Set up paths
        input_image = "input/input_image.jpg"
        known_faces_dir = "reference_library/"
        output_path = "output/analyzed_image.jpg"

        logger.info(f"Starting analysis of image: {input_image}", extra=extra)

        # First try to identify any known people
        logger.info("Checking for known people...", extra=extra)
        face_results = analyzer.compare_faces_with_library(input_image, known_faces_dir)

        # Read image bytes once for all operations
        logger.debug("Reading image bytes", extra=extra)
        with open(input_image, "rb") as image_file:
            image_bytes = image_file.read()

        # Get general content analysis
        logger.info("Performing content analysis", extra=extra)
        content_results = analyzer.detect_labels_in_image(image_bytes)

        # Get detailed face analysis
        logger.info("Performing detailed face analysis", extra=extra)
        face_details = analyzer.detect_face_details(image_bytes)

        # Print comprehensive analysis results
        logger.info("=== Image Analysis Results ===", extra=extra)

        # 1. Print people detection results
        if content_results and content_results.get("people"):
            logger.info("\n👥 People Detection:", extra=extra)
            for person in content_results["people"]:
                logger.info(
                    f"- Person detected (Confidence: {person['confidence']})",
                    extra=extra,
                )

            # 2. Print detailed face attributes for each detected face
            if face_details:
                logger.info("\n👤 Face Details:", extra=extra)
                for i, face in enumerate(face_details, 1):
                    logger.info(f"\nFace {i}:", extra=extra)

                    # Age Range
                    if "AgeRange" in face:
                        logger.info(
                            f"  Age Range: {face['AgeRange']['Low']}-{face['AgeRange']['High']} years",
                            extra=extra,
                        )

                    # Gender
                    if "Gender" in face:
                        logger.info(
                            f"  Gender: {face['Gender']['Value']} "
                            f"(Confidence: {face['Gender']['Confidence']:.2f}%)",
                            extra=extra,
                        )

                    # Emotions - Show top 3 emotions
                    if "Emotions" in face:
                        logger.info("  Emotions:", extra=extra)
                        sorted_emotions = sorted(
                            face["Emotions"],
                            key=lambda x: x["Confidence"],
                            reverse=True,
                        )[:3]
                        for emotion in sorted_emotions:
                            logger.info(
                                f"    - {emotion['Type'].lower().capitalize()}: "
                                f"{emotion['Confidence']:.2f}%",
                                extra=extra,
                            )

                    # Facial Features
                    features = {
                        "Smile": "Smiling",
                        "EyesOpen": "Eyes open",
                        "MouthOpen": "Mouth open",
                        "Eyeglasses": "Wearing glasses",
                        "Sunglasses": "Wearing sunglasses",
                    }

                    logger.info("  Facial Features:", extra=extra)
                    for key, description in features.items():
                        if key in face:
                            value = face[key]["Value"]
                            confidence = face[key]["Confidence"]
                            logger.info(
                                f"    - {description}: {'Yes' if value else 'No'} "
                                f"({confidence:.2f}%)",
                                extra=extra,
                            )

                    # Quality metrics
                    if "Quality" in face:
                        logger.info("  Image Quality:", extra=extra)
                        logger.info(
                            f"    - Brightness: {face['Quality']['Brightness']:.2f}",
                            extra=extra,
                        )
                        logger.info(
                            f"    - Sharpness: {face['Quality']['Sharpness']:.2f}",
                            extra=extra,
                        )

        # 3. Print animal detection results
        if content_results and content_results.get("animals"):
            logger.info("\n🐾 Animals Detected:", extra=extra)
            for animal in content_results["animals"]:
                logger.info(
                    f"- {animal['name']} (Confidence: {animal['confidence']})",
                    extra=extra,
                )

        # 4. Print clothing and accessories
        if content_results and content_results.get("objects"):
            clothing_items = [
                obj
                for obj in content_results["objects"]
                if any(
                    item in obj["name"].lower()
                    for item in [
                        "clothing",
                        "shirt",
                        "pants",
                        "dress",
                        "jacket",
                        "hat",
                        "accessories",
                        "shoes",
                    ]
                )
            ]

            if clothing_items:
                logger.info("\n👔 Clothing and Accessories:", extra=extra)
                for item in clothing_items:
                    logger.info(
                        f"- {item['name']} (Confidence: {item['confidence']})",
                        extra=extra,
                    )

        # Draw boxes on the image
        logger.info("\nGenerating annotated image...", extra=extra)
        if face_results.get("matches") and any(
            match["similarity"] > 0 for match in face_results["matches"]
        ):
            # If known faces were found, draw face recognition boxes
            logger.info("Drawing identified faces boxes", extra=extra)
            analyzer.draw_identified_faces(
                input_image, face_results["matches"], output_path
            )
        else:
            # If no known faces, draw general detection boxes
            logger.info("Drawing general detection boxes", extra=extra)
            analyzer.draw_bounding_boxes(input_image, content_results, output_path)

        logger.info(f"Annotated image saved to: {output_path}", extra=extra)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True, extra=extra)
        raise


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Set up logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler("logs/brutus.log")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the filter to both handlers
    request_id_filter = RequestIdFilter()
    file_handler.addFilter(request_id_filter)
    console_handler.addFilter(request_id_filter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    main()
