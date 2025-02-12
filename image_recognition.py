"""Feature to return image attributes using AWS Rekognition"""
from pathlib import Path
import io
import os
import json
from logging_utils import setup_logger
import boto3
import botocore
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont, ExifTags


logger = setup_logger(__name__)

class ImageAnalyzer:
    def __init__(self):
        try:
            self.rekognition_client = boto3.client("rekognition")
            logger.info("Successfully initialized Rekognition client")
        except Exception:
            logger.error("Failed to initialize client", exc_info=True)
            raise

    @staticmethod
    def _source_image(image_path):
        """
        Load and return an image bytes with correct orientation based on EXIF data.
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
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)
                if orientation:
                    rotate_values = {
                        3: 180,
                        6: 270,
                        8: 90
                    }
                    if orientation in rotate_values:
                        image = image.rotate(rotate_values[orientation], expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        
        with io.BytesIO() as bio:
            image.save(bio, format='JPEG')
            return bio.getvalue()

    def detect_labels_in_image(self, image_path, min_confidence=90):
        """
        Detect labels (objects, events) in the image.

        Args:
            image_path (str): Path to the image file
            min_confidence (int): Minimum confidence percentage (0-100) for label detection

        Returns:
            dict: Results containing:
                - labels_found: number of labels detected
                - labels: list of detected labels with confidence scores
                - error: error message if detection fails
        """
        logger.info(
            "Starting label detection",
            extra={
                "params": {
                    "image_path": str(image_path),
                    "min_confidence": min_confidence
                }
            }
        )

        try:
            image_bytes = self._source_image(image_path)
            response = self.rekognition_client.detect_labels(
                Image={"Bytes": image_bytes},
                MinConfidence=min_confidence
            )

            processed_labels = []
            instance_counter = 1

            for label in response['Labels']:
                processed_label = {
                    'Name': label['Name'],
                    'Confidence': label['Confidence']
                }

                if 'Instances' in label and label['Instances']:
                    processed_instances = []
                    for instance in label['Instances']:
                        processed_instance = {
                            'BoundingBox': instance['BoundingBox'],
                            'Confidence': instance['Confidence'],
                            'label_number': instance_counter
                        }
                        processed_instances.append(processed_instance)
                        instance_counter += 1
                    processed_label['Instances'] = processed_instances
                else:
                    processed_label['Instances'] = []

                processed_labels.append(processed_label)

            return processed_labels

        except Exception as e:
            logger.error(f"Error detecting labels: {str(e)}")
            raise

        except botocore.exceptions.ClientError as e:
            error_details = {
                "error": {
                    "type": "AWS Error",
                    "message": str(e)
                }
            }
            logger.error("AWS Rekognition error", extra=error_details)
            return error_details

    def detect_and_return_face_details(self, image_path):
        """
        Detect face details in an image and return analysis.

        Args:
            image_path (str): Path to the image file to analyze

        Returns:
            dict: Structured face analysis containing:
                - faces_found: number of faces detected
                - analysis_details: 
                    dict with age_range, 
                    gender, 
                    primary_emotion
        """

        logger.info("Starting face detection")
        try:
            response = self.rekognition_client.detect_faces(
                Image={"Bytes": self._source_image(image_path)},
                Attributes=["ALL"]
            )

            faces = response.get("FaceDetails", [])

            face_analyses = []
            for face_number, face in enumerate(response['FaceDetails'], 1):
                emotions = face.get("Emotions", [])
                primary_emotion = emotions[0] if emotions else {}
                box = face['BoundingBox']

                face_analysis = {
                    "age_range": face.get("AgeRange", {}),
                    "gender": face.get("Gender", {}).get("Value"),
                    "primary_emotion": primary_emotion.get("Type"),
                    "emotion_confidence": primary_emotion.get("Confidence"),
                    'face_number': face_number,
                    'BoundingBox': box
                }
                face_analyses.append(face_analysis)

            result = {
                "faces_found": len(faces),
                "faces": face_analyses
            }
            
            logger.info(
            "Face detection completed",
            extra={"results": json.dumps(result)}
            )
            
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Face detection failed",
                extra={"error": error_msg}
            )
            return {"error": error_msg}

    def compare_faces_with_library(
            self, target_image_path, library_folder, similarity_threshold=80):
        """
        Compare faces in image with library of reference images.

        Args:
            target_image_path (str): Path to the target image file
            library_folder (str): Path to folder containing reference images
            similarity_threshold (int): Minimum similarity percentage (0-100) to consider a match

        Returns:
            dict: Results containing:
                - matches_found: number of people identified
                - matches: list of identified people with their confidence scores
        """
        logger.info(
            "Starting face comparison",
            extra={
                "target_image": str(target_image_path),
                "library_folder": str(library_folder),
                "similarity_threshold": similarity_threshold
            }
        )

        target_image_bytes = self._source_image(target_image_path)
        target_faces = self.rekognition_client.detect_faces(
            Image={"Bytes": target_image_bytes},
            Attributes=["DEFAULT"]
        )
        if not target_faces["FaceDetails"]:
            logger.info("No faces detected", 
                        extra={"image_path": str(target_image_path)})
            return {"error": "No faces detected"}

        best_matches = {}

        library_images = [f for f in os.listdir(library_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for library_image in library_images:
            library_image_path = os.path.join(library_folder, library_image)
            person_name = os.path.splitext(library_image)[0] # Extract person name from filename
            try:
                library_image_bytes = self._source_image(library_image_path)
                comparison = self.rekognition_client.compare_faces(
                    SourceImage={"Bytes": library_image_bytes},
                    TargetImage={"Bytes": target_image_bytes},
                    SimilarityThreshold=similarity_threshold
                )
                face_matches = comparison.get("FaceMatches", [])
                if face_matches:
                    best_match = max(face_matches, key=lambda x: x["Similarity"])
                    if person_name not in best_matches or \
                       best_match["Similarity"] > best_matches[person_name]["similarity"]:
                        best_matches[person_name] = {
                            "similarity": best_match["Similarity"],
                            "confidence": best_match["Face"]["Confidence"]
                        }
            except self.rekognition_client.exceptions.InvalidParameterException:
                logger.warning(
                "Invalid reference image",
                extra={"image_path": str(library_image_path)}
                )
                continue
        identified_people = [
            {
                "person": person_name,
                "similarity": match_info["similarity"],
                "confidence": match_info["confidence"]
            }
            for person_name, match_info in best_matches.items()
        ]
        result = {
            "matches_found": len(identified_people),
            "matches": identified_people
        }
        logger.info(
            "Face comparison completed",
            extra={"results": result}
        )
        return result

    def draw_bounding_boxes(self, image_path, known_faces_dir):
        """
        Draw bounding boxes on image for all detected objects and faces
        Gets detection results by calling class methods
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
        font = ImageFont.load_default()


        faces = self.detect_and_return_face_details(image_path)
        comparison_results = self.compare_faces_with_library(image_path, known_faces_dir)
        labels = self.detect_labels_in_image(image_path)

        # Check if there are any humans or animals in the image.
        has_humans = faces and "error" not in faces and 'faces' in faces and len(faces['faces']) > 0
        has_animals = any(
            label['Name'] in ['Animal', 'Pet', 'Dog', 'Cat', 'Bird'] 
            for label in labels if 'Instances' in label and label['Instances']
        )

        if has_humans:
            recognized_faces = {}
            if (comparison_results and "error" not in comparison_results and 
                'matches' in comparison_results and comparison_results['matches']):

                # For each recognized person, get their face match details and bounding box info
                for match in comparison_results['matches']:
                    source_image = os.path.join(known_faces_dir, f"{match['person']}.jpg")
                    try:
                        face_matches = self.rekognition_client.compare_faces(
                            SourceImage={'Bytes': self._source_image(source_image)},
                            TargetImage={'Bytes': self._source_image(image_path)},
                            SimilarityThreshold=80
                        )

                        if face_matches['FaceMatches']:
                            matched_face = face_matches['FaceMatches'][0]['Face']
                            recognized_faces[str(matched_face['BoundingBox'])] = match['person']
                    except Exception as e:
                        logger.error(f"Error comparing faces: {str(e)}")
                        continue

            # Draw boxes for all faces
            for face in faces['faces']:
                box = face['BoundingBox']
                box_str = str(box)

                left = width * box['Left']
                top = height * box['Top']
                right = left + (width * box['Width'])
                bottom = top + (height * box['Height'])

                if box_str in recognized_faces:
                    # Draw orange box for recognized face
                    draw.rectangle([left, top, right, bottom], 
                                 outline='orange', width=3)
                    # Add face number and name
                    text = f"{face['face_number']}-{recognized_faces[box_str]}"
                    draw.text((left, top-20), text, 
                             fill='orange', font=font)
                else:
                    draw.rectangle([left, top, right, bottom], 
                                 outline='red', width=2)
                    draw.text((left, top-20), str(face['face_number']), 
                             fill='red', font=font)

        # Handle animals if present
        if has_animals:
            for label in labels:
                if ('Instances' in label and 
                    label['Name'] in ['Animal', 'Pet', 'Dog', 'Cat', 'Bird']):
                    for instance in label['Instances']:
                        box = instance['BoundingBox']

                        left = width * box['Left']
                        top = height * box['Top']
                        right = left + (width * box['Width'])
                        bottom = top + (height * box['Height'])

                        # Draw blue box for animals
                        draw.rectangle([left, top, right, bottom], 
                                     outline='blue', width=2)
                        if 'label_number' in instance:
                            draw.text((left, top-20), str(instance['label_number']), 
                                    fill='blue', font=font)

        # Only draw other objects if no humans or animals are present
        if not has_humans and not has_animals:
            for label in labels:
                if 'Instances' in label:
                    for instance in label['Instances']:
                        box = instance['BoundingBox']

                        left = width * box['Left']
                        top = height * box['Top']
                        right = left + (width * box['Width'])
                        bottom = top + (height * box['Height'])

                        # Draw green box for other objects
                        draw.rectangle([left, top, right, bottom], 
                                     outline='green', width=2)
                        if 'label_number' in instance:
                            draw.text((left, top-20), str(instance['label_number']), 
                                    fill='green', font=font)

        return image
