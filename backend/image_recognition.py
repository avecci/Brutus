"""Feature to return image attributes using AWS Rekognition."""
import io
import json
import os
from typing import Any, Dict, List, Union

import boto3
import botocore
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from logging_utils import setup_logger

logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()


class BrutusEyes:
    """Class to acts as Brutus' eyes. Analyzes given image using AWS Rekognition.

    Returns:
     - labels for objects
     - human features
     - facial recognition if image contains a recognised person such as Master.
    """

    def __init__(self) -> None:
        """Initialize Rekognition client or die trying."""
        try:
            profile_name = os.getenv("AWS_PROFILE")
            region_name = os.getenv("AWS_REGION", "eu-central-1")
            if not profile_name:
                logger.error("AWS_PROFILE not found in .env file")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            self.rekognition_client = session.client("rekognition")
            logger.info("Successfully initialized Rekognition client")
            self.font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
        except Exception:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    @staticmethod
    def _source_image(image_path) -> bytes:
        """Load and return an image bytes with correct orientation based on EXIF data.

        This ensures consistent orientation whether image was taken in
        portrait or landscape mode.

        If orientation data (EXIF) does not exist, continue with original image.

        Args:
            image_path (str): Path to the source image file

        Returns:
            bytes: Image bytes with correct orientation
        """
        image = Image.open(image_path)

        try:
            exif = image.getexif()
            if exif is not None:
                orientation = exif.get(274)
                if orientation:
                    rotate_values = {3: 180, 6: 270, 8: 90}
                    if orientation in rotate_values:
                        image = image.rotate(rotate_values[orientation], expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        with io.BytesIO() as bio:
            image.save(bio, format="JPEG")
            return bio.getvalue()

    @staticmethod
    def _get_overlap_ratio(box1, box2) -> float:
        """Act as a helper method for detect_labels_in_image().

        Calculate intersection over union (IoU) of two bounding boxes.
        Assist detecting if detected labels are the same so
        label detection only returns the main label for detected object.

        Returns:
            Intersection over union ratio.
        """
        x1 = max(box1["Left"], box2["Left"])
        y1 = max(box1["Top"], box2["Top"])
        x2 = min(box1["Left"] + box1["Width"], box2["Left"] + box2["Width"])
        y2 = min(box1["Top"] + box1["Height"], box2["Top"] + box2["Height"])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = box1["Width"] * box1["Height"]
        box2_area = box2["Width"] * box2["Height"]
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    @staticmethod
    def _is_related_label(label1, label2) -> bool:
        """Act as a helper method for detect_labels_in_image() to check if labels are related through parent-child relationship.

        Returns:
            True if labels are related, False otherwise
        """
        for parent in label1.get("Parents", []):
            if parent["Name"] == label2["Name"]:
                return True
        for parent in label2.get("Parents", []):
            if parent["Name"] == label1["Name"]:
                return True
        return False

    def detect_labels_in_image(
        self, image_path, min_confidence=90
    ) -> List[Dict[str, Any]]:
        """
        Detect labels (objects, events) in the image and consolidate related labels.

        Returns consolidated labels to avoid redundancy.
        """
        logger.info(
            "Starting label detection for image", extra={"image_file": image_path}
        )

        try:
            image_bytes = self._source_image(image_path)
            response = self.rekognition_client.detect_labels(
                Image={"Bytes": image_bytes}, MinConfidence=min_confidence
            )

            # Consolidate labels
            consolidated = {}

            for label in response["Labels"]:
                name = label["Name"]
                confidence = label["Confidence"]
                parents = label.get("Parents", [])
                instances = label.get("Instances", [])

                # Create a key that combines the base concept and its parents
                base_concept = None
                if parents:
                    base_concept = parents[-1]["Name"]  # Use the most general parent
                else:
                    base_concept = name

                if base_concept not in consolidated:
                    consolidated[base_concept] = {
                        "Name": base_concept,
                        "Confidence": confidence,
                        "RelatedLabels": set(),
                        "Instances": [],  # Initialize as empty list
                        "Parents": parents,
                    }

                # Add the current label as related if it's not the base concept
                if name != base_concept:
                    consolidated[base_concept]["RelatedLabels"].add(name)

                # Update confidence if higher
                if confidence > consolidated[base_concept]["Confidence"]:
                    consolidated[base_concept]["Confidence"] = confidence

                # Add instances without duplication
                if instances:
                    instance_dict = {}
                    for instance in instances:
                        # Create a unique key for each instance based on its bounding box
                        key = tuple((k, v) for k, v in instance["BoundingBox"].items())
                        if key not in instance_dict:
                            instance_dict[key] = instance
                            instance["label_number"] = len(instance_dict)
                            instance_dict[key] = instance
                    consolidated[base_concept]["Instances"] = list(
                        instance_dict.values()
                    )

            # Convert sets to sorted lists for JSON serialization
            consolidated_labels = []
            for label in consolidated.values():
                label["RelatedLabels"] = sorted(label["RelatedLabels"])
                consolidated_labels.append(label)

            # Sort by confidence
            consolidated_labels.sort(key=lambda x: x["Confidence"], reverse=True)

            logger.info(
                "Labels detected and consolidated:",
                extra={
                    "Results": json.dumps(consolidated_labels),
                    "image_file": image_path,
                },
            )
            return consolidated_labels

        except Exception as e:
            logger.error(f"Error detecting labels: {str(e)}")
            raise

        except botocore.exceptions.ClientError as e:
            error_details = {"error": {"type": "AWS Error", "message": str(e)}}
            logger.error("AWS Rekognition error", extra=error_details)
            return error_details

    def detect_and_return_face_details(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect face details in an image and return analysis.

        Args:
            image_path (str): Path to the image file to analyze

        Returns:
            Dictionary containing:
                - faces_found: number of faces detected
                - faces: list of face details with age_range, gender, emotions, etc.
                - error: error message if detection fails
        """
        logger.info("Starting face detection")
        try:
            response = self.rekognition_client.detect_faces(
                Image={"Bytes": self._source_image(image_path)}, Attributes=["ALL"]
            )

            faces = response.get("FaceDetails", [])

            face_analyses = []
            for face_number, face in enumerate(response["FaceDetails"], 1):
                emotions = face.get("Emotions", [])
                primary_emotion = emotions[0] if emotions else {}
                box = face["BoundingBox"]

                characteristics = {
                    "eyeglasses": {
                        "value": face.get("Eyeglasses", {}).get("Value", False),
                        "confidence": face.get("Eyeglasses", {}).get("Confidence", 0.0),
                    },
                    "sunglasses": {
                        "value": face.get("Sunglasses", {}).get("Value", False),
                        "confidence": face.get("Sunglasses", {}).get("Confidence", 0.0),
                    },
                    "beard": {
                        "value": face.get("Beard", {}).get("Value", False),
                        "confidence": face.get("Beard", {}).get("Confidence", 0.0),
                    },
                    "mustache": {
                        "value": face.get("Mustache", {}).get("Value", False),
                        "confidence": face.get("Mustache", {}).get("Confidence", 0.0),
                    },
                    "eyes_open": {
                        "value": face.get("EyesOpen", {}).get("Value", False),
                        "confidence": face.get("EyesOpen", {}).get("Confidence", 0.0),
                    },
                    "mouth_open": {
                        "value": face.get("MouthOpen", {}).get("Value", False),
                        "confidence": face.get("MouthOpen", {}).get("Confidence", 0.0),
                    },
                    "smile": {
                        "value": face.get("Smile", {}).get("Value", False),
                        "confidence": face.get("Smile", {}).get("Confidence", 0.0),
                    },
                    "face_occluded": {
                        "value": face.get("FaceOccluded", {}).get("Value", False),
                        "confidence": face.get("FaceOccluded", {}).get(
                            "Confidence", 0.0
                        ),
                    },
                }

                face_analysis = {
                    "age_range": face.get("AgeRange", {}),
                    "gender": face.get("Gender", {}).get("Value"),
                    "primary_emotion": primary_emotion.get("Type"),
                    "emotion_confidence": primary_emotion.get("Confidence"),
                    "face_number": face_number,
                    "BoundingBox": box,
                    "characteristics": characteristics,
                }
                face_analyses.append(face_analysis)

            result = {"faces_found": len(faces), "faces": face_analyses}

            logger.info(
                "Face detection completed",
                extra={"image_file": image_path, "results": json.dumps(result)},
            )

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error("Face detection failed", extra={"error": error_msg})
            return {"error": error_msg}

    def compare_faces_with_library(
        self,
        target_image_path: str,
        library_folder: str,
        similarity_threshold: float = 85.0,
    ) -> Dict[str, Union[int, List[Dict[str, Union[str, float]]], str]]:
        """Compare faces in image with library of reference images.

        Args:
            target_image_path (str): Path to the target image file
            library_folder (str): Path to folder containing reference images
            similarity_threshold (int): Minimum similarity percentage (0-100) to consider a match

        Returns:
            dict: Results containing:
                - matches_found: number of people identified
                - matches: list of identified people with their confidence scores
                - error: error message if comparison fails
        """
        logger.info(
            "Starting face comparison",
            extra={
                "image_file": target_image_path,
                "library_folder": str(library_folder),
                "similarity_threshold": similarity_threshold,
            },
        )

        target_image_bytes = self._source_image(target_image_path)
        target_faces = self.rekognition_client.detect_faces(
            Image={"Bytes": target_image_bytes}, Attributes=["DEFAULT"]
        )
        if not target_faces["FaceDetails"]:
            logger.info(
                "No faces detected", extra={"image_file": str(target_image_path)}
            )
            return {"error": "No faces detected"}

        best_matches: Dict[str, Dict[str, float]] = {}

        library_images = [
            f
            for f in os.listdir(library_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for library_image in library_images:
            library_image_path = os.path.join(library_folder, library_image)
            person_name = os.path.splitext(library_image)[
                0
            ]  # Extract person name from filename
            try:
                library_image_bytes = self._source_image(library_image_path)
                comparison = self.rekognition_client.compare_faces(
                    SourceImage={"Bytes": library_image_bytes},
                    TargetImage={"Bytes": target_image_bytes},
                    SimilarityThreshold=similarity_threshold,
                )
                face_matches = comparison.get("FaceMatches", [])
                if face_matches:
                    best_match = max(face_matches, key=lambda x: x["Similarity"])
                    if (
                        person_name not in best_matches
                        or best_match["Similarity"]
                        > best_matches[person_name]["similarity"]
                    ):
                        best_matches[person_name] = {
                            "similarity": best_match["Similarity"],
                            "confidence": best_match["Face"]["Confidence"],
                        }
            except self.rekognition_client.exceptions.InvalidParameterException:
                logger.warning(
                    "Invalid reference image",
                    extra={"image_file": str(library_image_path)},
                )
                continue
        identified_people = [
            {
                "person": person_name,
                "similarity": match_info["similarity"],
                "confidence": match_info["confidence"],
            }
            for person_name, match_info in best_matches.items()
        ]
        result = {"matches_found": len(identified_people), "matches": identified_people}
        logger.info("Face comparison completed", extra={"results": result})
        return result

    def draw_bounding_boxes(self, image_path: str, known_faces_dir: str) -> Image.Image:
        """Draw bounding boxes on image for all detected objects and faces.

        Gets detection results by calling class methods.
        Colors:
            - Red for unrecognized faces
            - Orange for recognized faces
            - Blue for animals
            - Green for other objects (only if no humans/animals present)
        Args:
            image_path: Path to the image file
            known_faces_dir: Directory containing reference face images
        Returns:
            PIL Image with bounding boxes drawn
        """
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        font = ImageFont.truetype(self.font_path, size=24)

        faces = self.detect_and_return_face_details(image_path)
        comparison_results = self.compare_faces_with_library(
            image_path, known_faces_dir
        )
        labels = self.detect_labels_in_image(image_path)

        # Check if there are any humans or animals in the image.
        has_humans = (
            faces
            and "error" not in faces
            and "faces" in faces
            and len(faces["faces"]) > 0
        )
        has_animals = any(
            label["Name"] in ["Animal", "Pet", "Dog", "Cat", "Bird", "Bear", "Snake"]
            for label in labels
            if "Instances" in label and label["Instances"]
        )

        if has_humans:
            recognized_faces = {}
            if (
                comparison_results
                and "error" not in comparison_results
                and "matches" in comparison_results
                and comparison_results["matches"]
            ):
                # For each recognized person, get their face match details and bounding box info
                for match in comparison_results["matches"]:
                    source_image = os.path.join(
                        known_faces_dir, f"{match['person']}.jpg"
                    )
                    try:
                        face_matches = self.rekognition_client.compare_faces(
                            SourceImage={"Bytes": self._source_image(source_image)},
                            TargetImage={"Bytes": self._source_image(image_path)},
                            SimilarityThreshold=80,
                        )

                        if face_matches["FaceMatches"]:
                            matched_face = face_matches["FaceMatches"][0]["Face"]
                            recognized_faces[str(matched_face["BoundingBox"])] = match[
                                "person"
                            ]
                    except Exception as e:
                        logger.error(f"Error comparing faces: {str(e)}")
                        continue

            # Draw boxes for all faces
            for face in faces["faces"]:
                box = face["BoundingBox"]
                box_str = str(box)

                left = width * box["Left"]
                top = height * box["Top"]
                right = left + (width * box["Width"])
                bottom = top + (height * box["Height"])

                if box_str in recognized_faces:
                    # Draw orange box for recognized face
                    draw.rectangle(
                        [left, top, right, bottom], outline="orange", width=5
                    )
                    # Add face number and name
                    text = f"{face['face_number']}-{recognized_faces[box_str]}"
                    draw.text((left, top - 20), text, fill="orange", font=font)
                else:
                    draw.rectangle([left, top, right, bottom], outline="red", width=2)
                    # Add black background for text
                    text = str(face["face_number"])
                    text_bbox = draw.textbbox((left, top - 20), text, font=font)
                    draw.rectangle(
                        [
                            text_bbox[0] - 2,
                            text_bbox[1] - 2,
                            text_bbox[2] + 2,
                            text_bbox[3] + 2,
                        ],
                        fill="black",
                    )
                    draw.text((left, top - 20), text, fill="red", font=font)

        # Handle animals if present
        if has_animals:
            for label in labels:
                if "Instances" in label and label["Name"] in [
                    "Animal",
                    "Pet",
                    "Dog",
                    "Cat",
                    "Bird",
                    "Bear",
                    "Snake",
                ]:
                    for instance in label["Instances"]:
                        box = instance["BoundingBox"]

                        left = width * box["Left"]
                        top = height * box["Top"]
                        right = left + (width * box["Width"])
                        bottom = top + (height * box["Height"])

                        # Draw blue box for animals
                        draw.rectangle(
                            [left, top, right, bottom], outline="blue", width=2
                        )
                        if "label_number" in instance:
                            text = str(instance["label_number"])
                            text_bbox = draw.textbbox((left, top - 20), text, font=font)
                            draw.rectangle(
                                [
                                    text_bbox[0] - 2,
                                    text_bbox[1] - 2,
                                    text_bbox[2] + 2,
                                    text_bbox[3] + 2,
                                ],
                                fill="black",
                            )
                            draw.text((left, top - 20), text, fill="blue", font=font)

        # Only draw other objects if no humans or animals are present
        if not has_humans and not has_animals:
            for label in labels:
                if "Instances" in label:
                    for instance in label["Instances"]:
                        box = instance["BoundingBox"]

                        left = width * box["Left"]
                        top = height * box["Top"]
                        right = left + (width * box["Width"])
                        bottom = top + (height * box["Height"])

                        # Draw green box for other objects
                        draw.rectangle(
                            [left, top, right, bottom], outline="green", width=2
                        )
                        if "label_number" in instance:
                            text = str(instance["label_number"])
                            text_bbox = draw.textbbox((left, top - 20), text, font=font)
                            draw.rectangle(
                                [
                                    text_bbox[0] - 2,
                                    text_bbox[1] - 2,
                                    text_bbox[2] + 2,
                                    text_bbox[3] + 2,
                                ],
                                fill="black",
                            )
                            draw.text((left, top - 20), text, fill="green", font=font)

        return image
