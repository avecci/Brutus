import logging
from pythonjsonlogger.jsonlogger import JsonFormatter
import uuid
from pathlib import Path
from image_recognition import ImageAnalyzer, RequestIdFilter

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
                faces_data = []
                for i, face in enumerate(face_details, 1):
                    face_info = {"face_number": i, "attributes": {}}

                    # Age Range
                    if "AgeRange" in face:
                        face_info["attributes"]["age_range"] = {
                            "low": face["AgeRange"]["Low"],
                            "high": face["AgeRange"]["High"],
                        }

                    # Gender
                    if "Gender" in face:
                        face_info["attributes"]["gender"] = {
                            "value": face["Gender"]["Value"],
                            "confidence": f"{face['Gender']['Confidence']:.2f}%",
                        }

                    # Emotions - Include all emotions in a list
                    if "Emotions" in face:
                        sorted_emotions = sorted(
                            face["Emotions"],
                            key=lambda x: x["Confidence"],
                            reverse=True,
                        )[:3]
                        face_info["attributes"]["emotions"] = [
                            {
                                "type": emotion["Type"].lower().capitalize(),
                                "confidence": f"{emotion['Confidence']:.2f}%",
                            }
                            for emotion in sorted_emotions
                        ]

                    # Facial Features
                    features = {
                        "Smile": "smiling",
                        "EyesOpen": "eyes_open",
                        "MouthOpen": "mouth_open",
                        "Eyeglasses": "wearing_glasses",
                        "Sunglasses": "wearing_sunglasses",
                    }

                    face_info["attributes"]["facial_features"] = {}
                    for key, attr_name in features.items():
                        if key in face:
                            face_info["attributes"]["facial_features"][attr_name] = {
                                "value": "Yes" if face[key]["Value"] else "No",
                                "confidence": f"{face[key]['Confidence']:.2f}%",
                            }

                    faces_data.append(face_info)

                # Log all face data as a single structured record
                logger.info(
                    "Face Analysis Results",
                    extra={"request_id": request_id, "faces_data": faces_data},
                )
        # Similarly structure other data
        if content_results and content_results.get("people"):
            people_data = [
                {"confidence": person["confidence"]}
                for person in content_results["people"]
            ]
            logger.info(
                "People Detection Results",
                extra={"request_id": request_id, "people": people_data},
            )

        # 3. Animals
        if content_results and content_results.get("animals"):
            animals_data = [
                {"name": animal["name"], "confidence": animal["confidence"]}
                for animal in content_results["animals"]
            ]
            logger.info(
                "Animal Detection Results",
                extra={"request_id": request_id, "animals": animals_data},
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
                logger.info("\n Clothing and Accessories:", extra=extra)
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

    # Create JSON formatter
    class CustomJsonFormatter(JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record["timestamp"] = record.created
            log_record["level"] = record.levelname
            log_record["logger"] = record.name
            log_record["request_id"] = getattr(record, "request_id", "-")

            # Add any additional fields from extra
            if hasattr(record, "faces"):
                log_record["faces"] = record.faces
            if hasattr(record, "people"):
                log_record["people"] = record.people
            if hasattr(record, "animals"):
                log_record["animals"] = record.animals

    formatter = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(logger)s %(request_id)s %(message)s"
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
