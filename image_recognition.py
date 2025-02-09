"""Feature to return image attributes using AWS Rekognition"""
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger.json import JsonFormatter
from pathlib import Path
import uuid
import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont, ExifTags


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def setup_logger():
    """Configure logging with rotating file handler and console output in JSON format"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    Path("./logs").mkdir(exist_ok=True)

    # Create JSON formatter
    class CustomJsonFormatter(JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record["timestamp"] = record.created
            log_record["level"] = record.levelname
            log_record["logger"] = record.name

            # Copy all extra fields
            if hasattr(record, "__dict__"):
                for key, value in record.__dict__.items():
                    if key not in [
                        "args",
                        "asctime",
                        "created",
                        "exc_info",
                        "exc_text",
                        "filename",
                        "funcName",
                        "levelname",
                        "levelno",
                        "lineno",
                        "module",
                        "msecs",
                        "msg",
                        "name",
                        "pathname",
                        "process",
                        "processName",
                        "relativeCreated",
                        "stack_info",
                        "thread",
                        "threadName",
                    ]:
                        log_record[key] = value

    formatter = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(logger)s %(request_id)s %(message)s"
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
        except Exception:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    def analyze_image_file(self, image_path, max_labels=30, save_path=None):
        """Analyze a local image file"""
        request_id = str(uuid.uuid4())
        extra = {
            "request_id": request_id,
            "metadata": {
                "image_path": image_path,
                "max_labels": max_labels,
                "save_path": save_path,
            },
        }
        logger.info("Starting image analysis", extra=extra)

        try:
            with open(image_path, "rb") as image:
                image_bytes = image.read()
                results = self.detect_labels_in_image(image_bytes, max_labels)
            if save_path:
                logger.info(
                    "Drawing bounding boxes",
                    extra={"request_id": request_id, "output_path": save_path},
                )
                self.draw_bounding_boxes(image_path, results, save_path)

            logger.info(
                "Image analysis completed",
                extra={"request_id": request_id, "results": results},
            )
            return results
        except FileNotFoundError:
            logger.error(
                "File not found",
                extra={
                    "request_id": request_id,
                    "error": {"type": "FileNotFoundError", "path": image_path},
                },
            )
        except Exception as e:
            logger.error(
                "Error processing image",
                extra={
                    "request_id": request_id,
                    "error": {"type": type(e).__name__, "message": str(e)},
                },
                exc_info=True,
            )

    def detect_face_details(self, image_bytes, request_id=None):
        """
        Detect face details in an image.
        Response includes details about:
        - Age Range (min/max)
        - Smile (boolean + confidence)
        - Eyeglasses/Sunglasses
        - Facial Hair
        - Gender
        - Emotions (happy, sad, angry, etc.)
        - Eye Open/Closed
        - Mouth Open/Closed
        - Quality (brightness, sharpness)
        - Pose (pitch, roll, yaw)
        """
        request_id = request_id or str(uuid.uuid4())
        extra = {"request_id": request_id}

        logger.info("Starting face detection", extra=extra)

        try:
            response = self.rekognition_client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"],  # or ['DEFAULT'] for basic attributes
            )

            faces = response.get("FaceDetails", [])

            # Log results with structured format
            for idx, face in enumerate(faces):
                logger.info(
                    "Face detection completed",
                    extra={
                        "request_id": request_id,
                        "detection_results": {
                            "faces_found": len(faces),
                            "analysis_details": {
                                "age_range": face.get("AgeRange", {}),
                                "gender": face.get("Gender", {}).get("Value"),
                                "primary_emotion": face.get("Emotions", [{}])[0].get(
                                    "Type"
                                ),
                            },
                        },
                    },
                )

            return faces
        except Exception as e:
            logger.error(
                "Face detection failed",
                extra={
                    "request_id": request_id,
                    "error": {"type": type(e).__name__, "message": str(e)},
                },
                exc_info=True,
            )
            raise

    def compare_faces_with_library(
        self, target_image_path, library_folder, similarity_threshold=80
    ):
        """
        Compare faces in target image with library of reference images
        """
        request_id = str(uuid.uuid4())
        metadata = {
            "request_id": request_id,
            "comparison_params": {
                "target_image": target_image_path,
                "library_folder": library_folder,
                "similarity_threshold": similarity_threshold,
            },
        }
        logger.info("Starting face comparison", extra=metadata)

        try:
            with open(target_image_path, "rb") as target_file:
                target_bytes = target_file.read()

            face_response = self.rekognition_client.detect_faces(
                Image={"Bytes": target_bytes}, Attributes=["DEFAULT"]
            )

            if not face_response["FaceDetails"]:
                logger.info(
                    "No faces detected in target image",
                    extra={
                        "request_id": request_id,
                        "target_analysis": {
                            "image_path": target_image_path,
                            "faces_found": 0,
                        },
                    },
                )
                return {"error": "No faces detected in target image"}

            face_count = len(face_response["FaceDetails"])
            logger.info(
                "Faces found in target image",
                extra={
                    "request_id": request_id,
                    "target_analysis": {
                        "image_path": target_image_path,
                        "faces_found": face_count,
                    },
                },
            )

            matches = []
            for face_index, target_face in enumerate(face_response["FaceDetails"]):
                target_box = target_face["BoundingBox"]
                face_metadata = {
                    "request_id": request_id,
                    "face_analysis": {
                        "face_index": face_index + 1,
                        "total_faces": face_count,
                        "location": target_box,
                        "confidence": target_face["Confidence"],
                    },
                }
                logger.info("Analyzing face", extra=face_metadata)

                best_match = {
                    "face_number": face_index + 1,
                    "person_name": "Unknown",
                    "similarity": 0,
                    "confidence": target_face["Confidence"],
                    "location": target_box,
                }

                for ref_image_path in Path(library_folder).glob("*.jpg"):
                    person_name = ref_image_path.stem
                    comparison_metadata = {
                        "request_id": request_id,
                        "comparison_details": {
                            "face_index": face_index + 1,
                            "person_name": person_name,
                            "reference_image": str(ref_image_path),
                        },
                    }
                    logger.info(
                        "Comparing with reference image", extra=comparison_metadata
                    )

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
                            logger.info(
                                "No match found",
                                extra={
                                    "request_id": request_id,
                                    "comparison_result": {
                                        "face_index": face_index + 1,
                                        "person_name": person_name,
                                        "match_found": False,
                                    },
                                },
                            )
                            continue

                        for match in compare_response["FaceMatches"]:
                            similarity = match["Similarity"]
                            match_box = match["Face"]["BoundingBox"]

                            match_metadata = {
                                "request_id": request_id,
                                "match_details": {
                                    "face_index": face_index + 1,
                                    "person_name": person_name,
                                    "similarity": f"{similarity:.2f}%",
                                    "location": match_box,
                                    "confidence": f"{match['Face']['Confidence']:.2f}%",
                                },
                            }
                            logger.info("Match found", extra=match_metadata)

                            if similarity > best_match["similarity"]:
                                match_info = {
                                    "person_name": person_name,
                                    "similarity": similarity,
                                    "confidence": match["Face"]["Confidence"],
                                    "location": match_box,
                                }
                                best_match.update(match_info)
                                logger.info(
                                    "New best match found",
                                    extra={
                                        "request_id": request_id,
                                        "best_match": match_info,
                                    },
                                )

                    except Exception as e:
                        logger.error(
                            "Error in face comparison",
                            extra={
                                "request_id": request_id,
                                "error": {
                                    "type": type(e).__name__,
                                    "message": str(e),
                                    "details": {
                                        "face_index": face_index + 1,
                                        "person_name": person_name,
                                        "reference_image": str(ref_image_path),
                                    },
                                },
                            },
                            exc_info=True,
                        )
                        continue

                matches.append(best_match)

            results = {"matches": matches, "error": None}

            logger.info(
                "Face comparison completed",
                extra={
                    "request_id": request_id,
                    "comparison_summary": {
                        "total_faces_analyzed": face_count,
                        "matches_found": len(matches),
                        "results": results,
                    },
                },
            )
            return results

        except Exception as e:
            error_info = {
                "request_id": request_id,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "details": {
                        "target_image": target_image_path,
                        "library_folder": library_folder,
                    },
                },
            }
            logger.error("Face comparison failed", extra=error_info, exc_info=True)
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
        Draw boxes and labels around identified faces in the image
        Args:
            image_path: Path to the original image
            matches: List of face matches with person names and locations
            save_path: Path where to save the annotated image
        """
        request_id = str(uuid.uuid4())
        metadata = {
            "request_id": request_id,
            "drawing_params": {
                "input_image": image_path,
                "output_image": save_path,
                "faces_to_draw": len(matches),
            },
        }
        logger.info("Starting to draw identified faces", extra=metadata)

        try:
            # Open and prepare image
            image = Image.open(image_path)

            # Correct image orientation if needed
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = image._getexif()
                if exif is not None and orientation in exif:
                    logger.info(
                        "Correcting image orientation",
                        extra={
                            "request_id": request_id,
                            "orientation": {
                                "original": exif[orientation],
                                "image_path": image_path,
                            },
                        },
                    )
                    self._correct_image_orientation(image, exif[orientation])
            except Exception as e:
                logger.warning(
                    "Could not process EXIF data",
                    extra={
                        "request_id": request_id,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "image_path": image_path,
                        },
                    },
                )

            draw = ImageDraw.Draw(image)
            image_width, image_height = image.size
            font = ImageFont.load_default()

            for idx, match in enumerate(matches, 1):
                try:
                    # Extract face information
                    box = match["location"]
                    person_name = match["person_name"]
                    similarity = match.get("similarity", 0)
                    confidence = match.get("confidence", 0)

                    # Calculate pixel coordinates
                    left = int(box["Left"] * image_width)
                    top = int(box["Top"] * image_height)
                    width = int(box["Width"] * image_width)
                    height = int(box["Height"] * image_height)

                    # Draw rectangle
                    draw.rectangle(
                        [(left, top), (left + width, top + height)],
                        outline="green",
                        width=2,
                    )

                    # Prepare label text
                    label = f"{person_name} ({similarity:.1f}%)"

                    # Draw label background and text
                    text_bbox = draw.textbbox((left, top - 25), label, font=font)
                    draw.rectangle([text_bbox], fill="green")
                    draw.text((left, top - 25), label, fill="white", font=font)

                    logger.info(
                        "Drew face box and label",
                        extra={
                            "request_id": request_id,
                            "face_drawing": {
                                "index": idx,
                                "person_name": person_name,
                                "similarity": f"{similarity:.1f}%",
                                "confidence": f"{confidence:.1f}%",
                                "location": {
                                    "left": left,
                                    "top": top,
                                    "width": width,
                                    "height": height,
                                },
                            },
                        },
                    )

                except Exception as e:
                    logger.error(
                        "Error drawing individual face",
                        extra={
                            "request_id": request_id,
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                                "face_index": idx,
                                "person_name": person_name,
                            },
                        },
                        exc_info=True,
                    )
                    continue

            # Save the annotated image
            try:
                image.save(save_path)
                logger.info(
                    "Successfully saved annotated image",
                    extra={
                        "request_id": request_id,
                        "save_details": {
                            "output_path": save_path,
                            "faces_drawn": len(matches),
                            "image_size": {
                                "width": image_width,
                                "height": image_height,
                            },
                        },
                    },
                )

            except Exception as e:
                logger.error(
                    "Failed to save annotated image",
                    extra={
                        "request_id": request_id,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "output_path": save_path,
                        },
                    },
                    exc_info=True,
                )
                raise

        except Exception as e:
            logger.error(
                "Failed to draw identified faces",
                extra={
                    "request_id": request_id,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "input_image": image_path,
                        "output_image": save_path,
                    },
                },
                exc_info=True,
            )
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

    def draw_bounding_boxes(self, image_path, results, save_path):
        """
        Draw bounding boxes on the image for detected objects
        Args:
            image_path: Path to the original image
            results: Dictionary containing detection results
            save_path: Path where to save the annotated image
        """
        request_id = str(uuid.uuid4())
        metadata = {
            "request_id": request_id,
            "drawing_params": {
                "input_image": image_path,
                "output_image": save_path,
                "detection_counts": {
                    "people": len(results.get("people", [])),
                    "animals": len(results.get("animals", [])),
                    "objects": len(results.get("objects", [])),
                },
            },
        }
        logger.info("Starting to draw bounding boxes", extra=metadata)

        try:
            # Open and prepare image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            width, height = image.size
            font = ImageFont.load_default()

            # Draw boxes for people (red)
            for person in results.get("people", []):
                if "bounding_box" in person and person["bounding_box"]:
                    box = person["bounding_box"]
                    self._draw_single_box(
                        draw,
                        box,
                        width,
                        height,
                        "Person",
                        person["confidence"],
                        "red",
                        font,
                        request_id,
                    )

            # Draw boxes for animals (blue)
            for animal in results.get("animals", []):
                if "bounding_box" in animal and animal["bounding_box"]:
                    box = animal["bounding_box"]
                    self._draw_single_box(
                        draw,
                        box,
                        width,
                        height,
                        animal["name"],
                        animal["confidence"],
                        "blue",
                        font,
                        request_id,
                    )

            # Draw boxes for other objects (green)
            for obj in results.get("objects", []):
                if "bounding_box" in obj and obj["bounding_box"]:
                    box = obj["bounding_box"]
                    self._draw_single_box(
                        draw,
                        box,
                        width,
                        height,
                        obj["name"],
                        obj["confidence"],
                        "green",
                        font,
                        request_id,
                    )

            # Save the annotated image
            image.save(save_path)
            logger.info(
                "Successfully saved annotated image",
                extra={
                    "request_id": request_id,
                    "save_details": {
                        "output_path": save_path,
                        "image_size": {"width": width, "height": height},
                    },
                },
            )

        except Exception as e:
            logger.error(
                "Failed to draw bounding boxes",
                extra={
                    "request_id": request_id,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "input_image": image_path,
                        "output_image": save_path,
                    },
                },
                exc_info=True,
            )
            raise

    def _draw_single_box(
        self,
        draw,
        box,
        image_width,
        image_height,
        label,
        confidence,
        color,
        font,
        request_id,
    ):
        """Helper method to draw a single bounding box with label"""
        try:
            # Ensure box coordinates are valid
            if not box or not all(
                key in box for key in ["Left", "Top", "Width", "Height"]
            ):
                logger.warning(
                    "Invalid box format",
                    extra={"request_id": request_id, "box_data": box},
                )
                return

            # Calculate pixel coordinates with bounds checking
            left = max(0, int(box["Left"] * image_width))
            top = max(0, int(box["Top"] * image_height))
            width = min(int(box["Width"] * image_width), image_width - left)
            height = min(int(box["Height"] * image_height), image_height - top)
            right = left + width
            bottom = top + height

            # Ensure coordinates are valid
            if left >= right or top >= bottom:
                logger.warning(
                    "Invalid box dimensions",
                    extra={
                        "request_id": request_id,
                        "dimensions": {
                            "left": left,
                            "right": right,
                            "top": top,
                            "bottom": bottom,
                        },
                    },
                )
                return

            # Draw rectangle
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=2)

            # Prepare label text
            label_text = f"{label} ({confidence:.1f}%)"

            # Calculate text position (ensure it's within image bounds)
            text_y = max(
                0, top - 25
            )  # Move text up by 25 pixels, but not outside image

            # Draw label background and text
            text_bbox = draw.textbbox((left, text_y), label_text, font=font)
            # Ensure text background stays within image bounds
            text_bbox = list(text_bbox)  # Convert to list to modify
            text_bbox[0] = max(0, text_bbox[0])  # x1
            text_bbox[1] = max(0, text_bbox[1])  # y1
            text_bbox[2] = min(image_width, text_bbox[2])  # x2
            text_bbox[3] = min(image_height, text_bbox[3])  # y2

            draw.rectangle(text_bbox, fill=color)
            draw.text((left, text_y), label_text, fill="white", font=font)

            logger.debug(
                "Successfully drew box",
                extra={
                    "request_id": request_id,
                    "box_details": {
                        "label": label,
                        "confidence": f"{confidence:.1f}%",
                        "position": {
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height,
                        },
                    },
                },
            )

        except Exception as e:
            logger.error(
                "Error drawing single box",
                extra={
                    "request_id": request_id,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "box": box,
                        "label": label,
                        "dimensions": {
                            "image_width": image_width,
                            "image_height": image_height,
                        },
                    },
                },
                exc_info=True,
            )

    def detect_labels_in_image(self, image_bytes, max_labels=30):
        """
        Detect labels in the image using AWS Rekognition
        Args:
            image_bytes: Binary image data
            max_labels: Maximum number of labels to return
        Returns:
            Dictionary containing detected labels and objects
        """
        request_id = str(uuid.uuid4())
        metadata = {
            "request_id": request_id,
            "detection_params": {
                "max_labels": max_labels,
                "image_size_bytes": len(image_bytes),
            },
        }
        logger.info("Starting label detection", extra=metadata)

        try:
            # First detect faces to get accurate person count and locations
            face_response = self.rekognition_client.detect_faces(
                Image={"Bytes": image_bytes}, Attributes=["DEFAULT"]
            )

            # Then get general labels
            label_response = self.rekognition_client.detect_labels(
                Image={"Bytes": image_bytes},
                MaxLabels=max_labels,
                Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"],
            )

            # Process the responses
            results = {
                "people": [],
                "animals": [],
                "objects": [],
                "image_properties": {},
            }

            # Add face detections as people
            for face in face_response.get("FaceDetails", []):
                if face.get("Confidence", 0) > 80:  # Confidence threshold
                    results["people"].append(
                        {
                            "name": "Person",
                            "confidence": face["Confidence"],
                            "bounding_box": face["BoundingBox"],
                        }
                    )

            # Process general labels
            for label in label_response.get("Labels", []):
                label_name = label["Name"]

                # Skip person labels as we already handled faces
                if label_name.lower() == "person":
                    continue

                # Categorize the label
                category = "objects"  # Default category
                if any(
                    animal in label_name.lower()
                    for animal in ["animal", "pet", "dog", "cat"]
                ):
                    category = "animals"

                # Add instances with bounding boxes
                for instance in label.get("Instances", []):
                    if instance.get("Confidence", 0) > 80:  # Confidence threshold
                        results[category].append(
                            {
                                "name": label_name,
                                "confidence": instance["Confidence"],
                                "bounding_box": instance["BoundingBox"],
                            }
                        )

            logger.info(
                "Label detection completed",
                extra={
                    "request_id": request_id,
                    "detection_results": {
                        "people_count": len(results["people"]),
                        "animal_count": len(results["animals"]),
                        "object_count": len(results["objects"]),
                    },
                },
            )

            return results

        except Exception as e:
            logger.error(
                "Error in label detection",
                extra={
                    "request_id": request_id,
                    "error": {"type": type(e).__name__, "message": str(e)},
                },
                exc_info=True,
            )
            raise

    def _process_response(self, response):
        """
        Process the response from Rekognition API
        Args:
            response: Raw response from Rekognition API
        Returns:
            Processed results dictionary
        Raises:
            ValueError: If response format is invalid
        """
        request_id = str(uuid.uuid4())

        try:
            # Validate response structure
            if "Labels" not in response:
                raise ValueError("Invalid response format: 'Labels' key missing")

            logger.debug(
                "Processing Rekognition response",
                extra={
                    "request_id": request_id,
                    "response_metadata": {
                        "label_count": len(response.get("Labels", [])),
                        "has_image_properties": "ImageProperties" in response,
                    },
                },
            )

            results = {
                "people": [],
                "animals": [],
                "objects": [],
                "image_properties": {},
            }

            # Process labels
            for label in response["Labels"]:  # Will raise KeyError if Labels is missing
                if (
                    not isinstance(label, dict)
                    or "Name" not in label
                    or "Confidence" not in label
                ):
                    raise ValueError("Invalid label format in response")

                label_info = {
                    "name": label["Name"],
                    "confidence": label["Confidence"],
                    "bounding_box": label.get("Instances", [{}])[0].get(
                        "BoundingBox", {}
                    )
                    if label.get("Instances")
                    else {},
                }

                # Categorize the label
                if label["Name"] == "Person":
                    results["people"].append(label_info)
                elif label["Name"] in ["Animal", "Pet", "Dog", "Cat"]:
                    results["animals"].append(label_info)
                else:
                    results["objects"].append(label_info)

            logger.debug(
                "Processed labels categorized",
                extra={
                    "request_id": request_id,
                    "categorization_results": {
                        "people_count": len(results["people"]),
                        "animals_count": len(results["animals"]),
                        "objects_count": len(results["objects"]),
                    },
                },
            )

            return results

        except Exception as e:
            logger.error(
                "Error processing Rekognition response",
                extra={
                    "request_id": request_id,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "response_keys": list(response.keys()),
                    },
                },
                exc_info=True,
            )
            raise
