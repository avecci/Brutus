"""Feature to return image attributes using AWS Rekognition"""
# import base64
# import os
import logging
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


logging.basicConfig(filename="./logs/brutus.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAnalyzer:
    def __init__(self):
        try:
            self.rekognition_client = boto3.client("rekognition")
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            raise

    def analyze_image_file(self, image_path, max_labels=20, save_path=None):
        """
        Analyze a local image file
        """
        try:
            with open(image_path, "rb") as image:
                image_bytes = image.read()
                results = self.detect_labels_in_image(image_bytes, max_labels)

                # If save_path is provided, draw bounding boxes and save the image
                if save_path:
                    self.draw_bounding_boxes(image_path, results, save_path)

                return results
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")

    def compare_faces_with_library(
        self, target_image_path, library_folder, similarity_threshold=80
    ):  # Lower threshold
        try:
            print(f"\nProcessing target image: {target_image_path}")
            with open(target_image_path, "rb") as target_file:
                target_bytes = target_file.read()

            face_response = self.rekognition_client.detect_faces(
                Image={"Bytes": target_bytes}, Attributes=["DEFAULT"]
            )

            if not face_response["FaceDetails"]:
                print("No faces detected in target image")
                return {"error": "No faces detected in target image"}

            matches = []

            for face_index, target_face in enumerate(face_response["FaceDetails"]):
                target_box = target_face["BoundingBox"]
                print(f"\nAnalyzing face {face_index + 1} in target image")
                print(f"Face location: {target_box}")

                best_match = {
                    "face_number": face_index + 1,
                    "person_name": "Unknown",
                    "similarity": 0,
                    "confidence": target_face["Confidence"],
                    "location": target_box,
                }

                for ref_image_path in Path(library_folder).glob("*.jpg"):
                    person_name = ref_image_path.stem
                    print(f"\nComparing with: {person_name}")

                    try:
                        with open(ref_image_path, "rb") as ref_file:
                            ref_bytes = ref_file.read()

                        compare_response = self.rekognition_client.compare_faces(
                            SourceImage={"Bytes": ref_bytes},
                            TargetImage={"Bytes": target_bytes},
                            SimilarityThreshold=similarity_threshold,
                            QualityFilter="LOW",  # More lenient quality filter
                        )

                        if not compare_response.get("FaceMatches"):
                            print(f"No match found with {person_name}")
                            continue

                        for match in compare_response["FaceMatches"]:
                            similarity = match["Similarity"]
                            match_box = match["Face"]["BoundingBox"]
                            print(f"Potential match found:")
                            print(f"- Similarity: {similarity:.2f}%")
                            print(f"- Match location: {match_box}")

                            if similarity > best_match["similarity"]:
                                best_match = {
                                    "face_number": face_index + 1,
                                    "person_name": person_name,
                                    "similarity": similarity,
                                    "confidence": match["Face"]["Confidence"],
                                    "location": match_box,
                                }
                                print(
                                    f"New best match: {person_name} with {similarity:.2f}% similarity"
                                )

                    except Exception as e:
                        print(f"Error comparing with {person_name}: {str(e)}")
                        continue

                matches.append(best_match)

            return {"matches": matches, "error": None}

        except Exception as e:
            print(f"Error in comparison process: {str(e)}")
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
        Draw bounding boxes and labels for identified faces with improved visualization
        """
        try:
            # Open image
            image = Image.open(image_path)
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

    def draw_bounding_boxes(self, original_image_path, results, save_path):
        """
        Draw bounding boxes on the image and save it
        Only draws boxes for people and animals
        """
        try:
            # Open the original image
            image = Image.open(original_image_path)
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
        Process and format the detection response
        """
        results = {
            "objects": [],
            "people": [],
            "animals": [],
            "other": [],
            "error": None,
        }

        try:
            for label in response["Labels"]:
                item = {
                    "name": label["Name"],
                    "confidence": f"{label['Confidence']:.2f}%",
                    "instances": [],
                }

                # Add bounding box information if available
                if label["Instances"]:
                    for instance in label["Instances"]:
                        box = instance["BoundingBox"]
                        item["instances"].append(
                            {"confidence": f"{instance['Confidence']:.2f}%", "box": box}
                        )

                # Categorize the label
                if label["Name"].lower() == "person":
                    results["people"].append(item)
                elif any(p.get("Name") == "Animal" for p in label.get("Parents", [])):
                    results["animals"].append(item)
                elif any(p.get("Name") == "Person" for p in label.get("Parents", [])):
                    results["people"].append(item)
                else:
                    results["objects"].append(item)

            return results
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")


def main():
    try:
        analyzer = ImageAnalyzer()

        # Set up paths
        input_image = "input/input_image.jpg"
        known_faces_dir = "reference_library/"
        output_path = "output/analyzed_image.jpg"

        print(f"\nAnalyzing image: {input_image}")

        # First try to identify any known people
        print("\nChecking for known people...")
        face_results = analyzer.compare_faces_with_library(input_image, known_faces_dir)

        if face_results.get("error"):
            print(f"Error in face comparison: {face_results['error']}")
            known_faces_found = False
        else:
            known_faces_found = any(
                match["similarity"] > 0 for match in face_results["matches"]
            )

            # Print face recognition results
            print("\nFace Recognition Results:")
            for match in face_results["matches"]:
                if match["similarity"] > 0:
                    print(f"\nFace {match['face_number']}:")
                    print(f"✓ Identified as: {match['person_name']}")
                    print(f"  Similarity: {match['similarity']}")
                    print(f"  Confidence: {match['confidence']}")

                    # Add confidence level indicator
                    if isinstance(match["similarity"], str):
                        similarity = float(match["similarity"].rstrip("%"))
                    else:
                        similarity = match["similarity"]

                    if similarity >= 95:
                        print("  Confidence Level: High ★★★")
                    elif similarity >= 90:
                        print("  Confidence Level: Medium ★★")
                    else:
                        print("  Confidence Level: Low ★")

        # Now analyze the general content of the image
        print("\nAnalyzing image content...")
        with open(input_image, "rb") as image_file:
            image_bytes = image_file.read()

        content_results = analyzer.detect_labels_in_image(image_bytes)

        if content_results.get("error"):
            print(f"Error in image analysis: {content_results['error']}")
        else:
            print("\nImage Content Analysis:")

            # Print people detected (even if not identified)
            if content_results.get("people"):
                print("\nPeople detected:")
                for person in content_results["people"]:
                    print(f"- Person detected (Confidence: {person['confidence']})")

            # Print animals detected
            if content_results.get("animals"):
                print("\nAnimals detected:")
                for animal in content_results["animals"]:
                    print(f"- {animal['name']} (Confidence: {animal['confidence']})")

            # Print other significant objects
            if content_results.get("objects"):
                print("\nOther objects detected:")
                for obj in content_results["objects"]:
                    print(f"- {obj['name']} (Confidence: {obj['confidence']})")

        # Draw boxes on the image
        print("\nGenerating annotated image...")
        if known_faces_found:
            # If known faces were found, draw face recognition boxes
            analyzer.draw_identified_faces(
                input_image, face_results["matches"], output_path
            )
        else:
            # If no known faces, draw general detection boxes
            analyzer.draw_bounding_boxes(input_image, content_results, output_path)

        print(f"Annotated image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
