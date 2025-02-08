"""Feature to return image attributes using AWS Rekognition"""
import boto3
import base64
import os
from botocore.exceptions import ClientError
import logging
from PIL import Image, ImageDraw


logging.basicConfig(filename='./logs/brutus.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        try:
            self.rekognition_client = boto3.client('rekognition')
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            raise

    def analyze_image_file(self, image_path, max_labels=20, save_path=None):
        """
        Analyze a local image file
        """
        try:
            with open(image_path, 'rb') as image:
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

    def draw_bounding_boxes(self, original_image_path, results, save_path):
        """
        Draw bounding boxes on the image and save it
        """
        try:
            # Open the original image
            image = Image.open(original_image_path)
            draw = ImageDraw.Draw(image)
            
            # Get image dimensions
            width, height = image.size
            
            # Define colors for different categories
            colors = {
                'people': (255, 0, 0),    # Red for people
                'animals': (0, 255, 0),   # Green for animals
                'objects': (0, 0, 255),   # Blue for objects
                'other': (255, 255, 0)    # Yellow for other
            }

            # Draw boxes for each category
            for category, items in results.items():
                if category != 'error' and items:
                    color = colors.get(category, (255, 255, 255))
                    
                    for item in items:
                        for instance in item['instances']:
                            box = instance['box']
                            
                            # Convert normalized coordinates to pixel values
                            left = int(box['Left'] * width)
                            top = int(box['Top'] * height)
                            right = int((box['Left'] + box['Width']) * width)
                            bottom = int((box['Top'] + box['Height']) * height)
                            
                            # Draw rectangle
                            draw.rectangle([left, top, right, bottom], 
                                         outline=color, width=3)
                            
                            # Draw label
                            #label = f"{item['name']} ({instance['confidence']})"
                            #draw.text((left, top-20), label, fill=color)

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
                Image={'Bytes': image_bytes},
                MaxLabels=max_labels
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
            'objects': [],
            'people': [],
            'animals': [],
            'other': [],
            'error': None
        }

        try:
            for label in response['Labels']:
                item = {
                    'name': label['Name'],
                    'confidence': f"{label['Confidence']:.2f}%",
                    'instances': []
                }

                # Add bounding box information if available
                if label['Instances']:
                    for instance in label['Instances']:
                        box = instance['BoundingBox']
                        item['instances'].append({
                            'confidence': f"{instance['Confidence']:.2f}%",
                            'box': box
                        })

                # Categorize the label
                if label['Name'].lower() == 'person':
                    results['people'].append(item)
                elif any(p.get('Name') == 'Animal' for p in label.get('Parents', [])):
                    results['animals'].append(item)
                elif any(p.get('Name') == 'Person' for p in label.get('Parents', [])):
                    results['people'].append(item)
                else:
                    results['objects'].append(item)

            return results
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")



def main():
    try:
        analyzer = ImageAnalyzer()
    
        image_path = './tests/pictures/one_person.jpg'
        output_path = './output/output.png'
        print(f"\nAnalyzing image: {image_path}")
        local_result = analyzer.analyze_image_file(image_path, save_path=output_path)

        # Check for errors first
        if local_result.get('error'):
            print(f"Error: {local_result['error']}")
            return
    
        print("\nAnalysis Results:")
        found_items = False
        for category, items in local_result.items():
            if items and category != 'error':
                found_items = True
                print(f"\n{category.title()}:")
                for item in items:
                    print(f"- {item['name']} (Confidence: {item['confidence']})")
                    if item['instances']:
                        print("  Bounding Box Locations:")
                        for instance in item['instances']:
                            box = instance['box']
                            print(f"    Confidence: {instance['confidence']}")
                            print(f"    Left: {box['Left']:.3f}")
                            print(f"    Top: {box['Top']:.3f}")
                            print(f"    Width: {box['Width']:.3f}")
                            print(f"    Height: {box['Height']:.3f}")
        
        if not found_items:
            print("No objects, people, or animals were detected in the image.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
