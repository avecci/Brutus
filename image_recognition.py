"""Feature to return image attributes using AWS Rekognition"""
import logging
from logging.handlers import RotatingFileHandler
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ExifTags
import uuid


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def setup_logger():
    """Configure logging with rotating file handler and console output"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    Path("./logs").mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s"
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        "./logs/brutus.log", maxBytes=1024 * 1024, backupCount=3  # 1MB
    )
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

    return logger


logger = setup_logger()


class ImageAnalyzer:
    def __init__(self):
        try:
            self.rekognition_client = boto3.client("rekognition")
            logger.info("Successfully initialized Rekognition client")
        except Exception as e:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    def analyze_image_file(self, image_path, max_labels=20, save_path=None):
        """Analyze a local image file"""
        request_id = str(uuid.uuid4())
        extra = {"request_id": request_id}
        logger.info(f"Starting image analysis for {image_path}", extra=extra)

        try:
            with open(image_path, "rb") as image:
                image_bytes = image.read()
                results = self.detect_labels_in_image(image_bytes, max_labels)
            if save_path:
                logger.info(f"Drawing bounding boxes to {save_path}", extra=extra)
                self.draw_bounding_boxes(image_path, results, save_path)

            logger.info("Image analysis completed successfully", extra=extra)
            return results
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)

    def detect_face_details(self, image_bytes, request_id=None):
        """
        # Response includes details about:
        # - Age Range (min/max)
        # - Smile (boolean + confidence)
        # - Eyeglasses/Sunglasses
        # - Facial Hair
        # - Gender
        # - Emotions (happy, sad, angry, etc.)
        # - Eye Open/Closed
        # - Mouth Open/Closed
        # - Quality (brightness, sharpness)
        # - Pose (pitch, roll, yaw)
        """
        extra = {"request_id": request_id or str(uuid.uuid4())}
        logger.info("Starting face detection", extra=extra)

        try:
            response = self.rekognition_client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"],  # or ['DEFAULT'] for basic attributes
            )

            face_count = len(response.get("FaceDetails", []))
            logger.info(f"Detected {face_count} faces", extra=extra)
            return response["FaceDetails"]
        except Exception as e:
            logger.error(
                f"Error detecting face details: {str(e)}", exc_info=True, extra=extra
            )

    def compare_faces_with_library(
        self, target_image_path, library_folder, similarity_threshold=80
    ):
        request_id = str(uuid.uuid4())
        extra = {"request_id": request_id}
        logger.info(
            f"Starting face comparison - Target: {target_image_path}, Library: {library_folder}",
            extra=extra,
        )
        try:
            with open(target_image_path, "rb") as target_file:
                target_bytes = target_file.read()

            face_response = self.rekognition_client.detect_faces(
                Image={"Bytes": target_bytes}, Attributes=["DEFAULT"]
            )

            if not face_response["FaceDetails"]:
                logger.info("No faces detected in target image", extra=extra)
                return {"error": "No faces detected in target image"}

            matches = []
            face_count = len(face_response["FaceDetails"])
            logger.info(f"Found {face_count} faces in target image", extra=extra)

            for face_index, target_face in enumerate(face_response["FaceDetails"]):
                target_box = target_face["BoundingBox"]
                logger.debug(
                    f"Analyzing face {face_index + 1} - Location: {target_box}",
                    extra=extra,
                )

                best_match = {
                    "face_number": face_index + 1,
                    "person_name": "Unknown",
                    "similarity": 0,
                    "confidence": target_face["Confidence"],
                    "location": target_box,
                }

                for ref_image_path in Path(library_folder).glob("*.jpg"):
                    person_name = ref_image_path.stem
                    logger.debug(f"Comparing with: {person_name}", extra=extra)

                    try:
                        with open(ref_image_path, "rb") as ref_file:
                            ref_bytes = ref_file.read()

                        compare_response = self.rekognition_client.compare_faces(
                            SourceImage={"Bytes": ref_bytes},
                            TargetImage={"Bytes": target_bytes},
                            SimilarityThreshold=similarity_threshold,
                            QualityFilter="LOW",
                        )

                        if not compare_response.get("FaceMatches"):
                            logger.debug(
                                f"No match found with {person_name}", extra=extra
                            )
                            continue

                        for match in compare_response["FaceMatches"]:
                            similarity = match["Similarity"]
                            match_box = match["Face"]["BoundingBox"]
                            logger.debug(
                                f"Match found - Similarity: {similarity:.2f}%, Location: {match_box}",
                                extra=extra,
                            )

                            if similarity > best_match["similarity"]:
                                best_match.update(
                                    {
                                        "person_name": person_name,
                                        "similarity": similarity,
                                        "confidence": match["Face"]["Confidence"],
                                        "location": match_box,
                                    }
                                )
                                logger.info(
                                    f"New best match: {person_name} with {similarity:.2f}% similarity",
                                    extra=extra,
                                )

                    except Exception as e:
                        logger.error(
                            f"Error comparing with {person_name}: {str(e)}",
                            exc_info=True,
                            extra=extra,
                        )
                        continue

                matches.append(best_match)

            logger.info("Face comparison completed successfully", extra=extra)
            return {"matches": matches, "error": None}

        except Exception as e:
            logger.error(
                f"Error in comparison process: {str(e)}", exc_info=True, extra=extra
            )
            return {"error": str(e)}

    def _is_same_face(self, box1, box2, tolerance=0.1):
        """
        Check if two bounding boxes refer to the same face

        Args:
            box1 (dict): First bounding box
            box2 (dict): Second bounding box
            tolerance (float): Maximum allowed difference in coordinates

        Returns:
            bool: True if boxes likely refer to same face
        """
        return (
            abs(box1["Left"] - box2["Left"]) < tolerance
            and abs(box1["Top"] - box2["Top"]) < tolerance
            and abs(box1["Width"] - box2["Width"]) < tolerance
            and abs(box1["Height"] - box2["Height"]) < tolerance
        )

    def draw_identified_faces(self, image_path, matches, save_path):
        """
        Draw bounding boxes and labels for identified faces
        """
        try:
            # Open and correct image orientation
            image = Image.open(image_path)
            image = self._correct_image_orientation(image)
            draw = ImageDraw.Draw(image)
            width, height = image.size

            # Try to load a font, fallback to default if necessary
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Draw box and label for each match
            for match in matches:
                box = match["location"]

                # Convert normalized coordinates to pixels
                left = int(box["Left"] * width)
                top = int(box["Top"] * height)
                right = int((box["Left"] + box["Width"]) * width)
                bottom = int((box["Top"] + box["Height"]) * height)

                # Choose color based on similarity score
                if isinstance(match["similarity"], str):
                    similarity = float(match["similarity"].rstrip("%"))
                else:
                    similarity = match["similarity"]

                if similarity >= 95:
                    color = (0, 255, 0)  # Green for high confidence
                elif similarity >= 90:
                    color = (255, 165, 0)  # Orange for medium confidence
                else:
                    color = (255, 0, 0)  # Red for lower confidence

                # Draw box
                draw.rectangle([left, top, right, bottom], outline=color, width=3)

                # Draw label with confidence
                if match["person_name"] != "Unknown":
                    label = f"{match['person_name']}\nSimilarity: {match['similarity']}"

                    # Create background for text
                    text_bbox = draw.textbbox((left, top - 45), label, font=font)
                    draw.rectangle(text_bbox, fill=color)

                    # Draw text in white
                    draw.text((left, top - 45), label, fill=(255, 255, 255), font=font)

            # Save result
            image.save(save_path)
            logger.info(f"Annotated image saved to: {save_path}")

            return True

        except Exception as e:
            logger.error(f"Error drawing identified faces: {str(e)}")
            raise

    def _correct_image_orientation(self, image):
        """
        Correct image orientation based on EXIF data
        """
        try:
            # Get EXIF data
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break

            exif = image._getexif()
            if exif is not None:
                if orientation in exif:
                    if exif[orientation] == 3:
                        image = image.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        image = image.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        image = image.rotate(90, expand=True)

            return image
        except (AttributeError, KeyError, IndexError, TypeError):
            # If there's no EXIF data or other issues, return original image
            return image

    def draw_bounding_boxes(self, original_image_path, results, save_path):
        """
        Draw bounding boxes on the image and save it
        """
        try:
            # Open and correct image orientation
            image = Image.open(original_image_path)
            image = self._correct_image_orientation(image)
            draw = ImageDraw.Draw(image)

            # Get image dimensions
            width, height = image.size

            # Define colors for different categories
            colors = {
                "people": (255, 0, 0),  # Red for people
                "animals": (0, 255, 0),  # Green for animals
            }

            # Draw boxes only for people and animals
            for category in ["people", "animals"]:
                if category in results and results[category]:
                    color = colors[category]

                    for item in results[category]:
                        for instance in item["instances"]:
                            box = instance["box"]

                            # Convert normalized coordinates to pixel values
                            left = int(box["Left"] * width)
                            top = int(box["Top"] * height)
                            right = int((box["Left"] + box["Width"]) * width)
                            bottom = int((box["Top"] + box["Height"]) * height)

                            draw.rectangle(
                                [left, top, right, bottom], outline=color, width=3
                            )

            # Save the image with bounding boxes
            image.save(save_path)
            logger.info(f"Image with bounding boxes saved to: {save_path}")

        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {str(e)}")
            raise

    def detect_labels_in_image(self, image_bytes, max_labels=20):
        """
        Detect labels in image bytes
        """
        try:
            response = self.rekognition_client.detect_labels(
                Image={"Bytes": image_bytes}, MaxLabels=max_labels
            )
            return self._process_response(response)
        except ClientError as e:
            logger.error(f"AWS Rekognition error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    def _process_response(self, response):
        """
        Process and format the detection response with deduplication
        """
        results = {
            "objects": [],
            "people": [],
            "animals": [],
            "other": [],
            "error": None,
        }

        try:
            # Keep track of processed boxes only for people
            processed_people_boxes = []

            for label in response["Labels"]:
                item = {
                    "name": label["Name"],
                    "confidence": f"{label['Confidence']:.2f}%",
                    "instances": [],
                }

                is_person = label["Name"].lower() == "person" or any(
                    p.get("Name") == "Person" for p in label.get("Parents", [])
                )

                # Add bounding box information if available
                if label["Instances"]:
                    for instance in label["Instances"]:
                        box = instance["BoundingBox"]

                        # Only apply deduplication to people
                        if is_person:
                            is_duplicate = False
                            for processed_box in processed_people_boxes:
                                if self._boxes_overlap(
                                    box, processed_box, threshold=0.5
                                ):
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                processed_people_boxes.append(box)
                                item["instances"].append(
                                    {
                                        "confidence": f"{instance['Confidence']:.2f}%",
                                        "box": box,
                                    }
                                )
                        else:
                            # For non-person objects, add all instances
                            item["instances"].append(
                                {
                                    "confidence": f"{instance['Confidence']:.2f}%",
                                    "box": box,
                                }
                            )

                # For objects without instances, or after processing instances
                if is_person:
                    if item["instances"]:  # Only add if we have unique instances
                        results["people"].append(item)
                elif any(p.get("Name") == "Animal" for p in label.get("Parents", [])):
                    results["animals"].append(item)
                else:
                    # For non-person objects, add them even without instances
                    if label["Instances"]:
                        results["objects"].append(item)
                    else:
                        # Add objects without bounding boxes
                        item_without_instances = {
                            "name": label["Name"],
                            "confidence": f"{label['Confidence']:.2f}%",
                        }
                        results["objects"].append(item_without_instances)

            return results
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")

    def _boxes_overlap(self, box1, box2, threshold=0.5):

        """
        Check if two bounding boxes overlap significantly

        Args:
            box1 (dict): First bounding box with Left, Top, Width, Height
            box2 (dict): Second bounding box with Left, Top, Width, Height
            threshold (float): Minimum intersection over union (IoU) to consider as overlap

        Returns:
            bool: True if boxes overlap significantly
        """
        # Calculate coordinates
        x1 = max(box1["Left"], box2["Left"])
        y1 = max(box1["Top"], box2["Top"])
        x2 = min(box1["Left"] + box1["Width"], box2["Left"] + box2["Width"])
        y2 = min(box1["Top"] + box1["Height"], box2["Top"] + box2["Height"])

        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = box1["Width"] * box1["Height"]
        area2 = box2["Width"] * box2["Height"]
        union = area1 + area2 - intersection

        # Calculate IoU (Intersection over Union)
        iou = intersection / union if union > 0 else 0

        return iou > threshold
