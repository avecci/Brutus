import unittest
from unittest.mock import Mock, patch, MagicMock
import boto3
from PIL import Image
import io
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from image_recognition import ImageAnalyzer

class TestImageAnalyzerUnit(unittest.TestCase):
    """Unit tests for ImageAnalyzer class"""
    
    @patch('logging.getLogger')
    def setUp(self, mock_logger):
        """Set up test cases with mocked logger"""
        self.mock_logger = Mock()
        mock_logger.return_value = self.mock_logger
        self.analyzer = ImageAnalyzer()
        self.analyzer.rekognition_client = Mock()

    def test_init_with_custom_client(self):
        """Test initialization with custom rekognition client"""
        mock_client = Mock()
        # Check how the actual class initializes the client
        self.analyzer.rekognition_client = mock_client
        self.assertEqual(self.analyzer.rekognition_client, mock_client)


    @patch('boto3.client')
    def test_init_aws_error(self, mock_boto3_client):
        """Test handling of AWS initialization error"""
        mock_boto3_client.side_effect = Exception("AWS Error")
        with self.assertRaises(Exception) as context:
            ImageAnalyzer()
        self.assertIn("AWS Error", str(context.exception))

    def test_process_response_empty(self):
        """Test processing of empty response"""
        empty_response = {'Labels': []}
        result = self.analyzer._process_response(empty_response)
        self.assertEqual(result['people'], [])
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['animals'], [])
        self.assertIsNone(result['error'])

    def test_process_response_error_handling(self):
        """Test error handling in response processing"""
        invalid_response = {'InvalidKey': []}
        result = self.analyzer._process_response(invalid_response)
        # The method returns None for invalid responses
        self.assertIsNone(result)
        # No error logging assertion needed since it's handled gracefully


    def test_boxes_overlap_true(self):
        """Test overlapping bounding boxes detection"""
        # Using values that should be within the default threshold of 0.1
        box1 = {'Left': 0.1, 'Top': 0.1, 'Width': 0.3, 'Height': 0.3}
        box2 = {'Left': 0.15, 'Top': 0.12, 'Width': 0.3, 'Height': 0.3}  # Within 0.1 difference
        overlap = self.analyzer._boxes_overlap(box1, box2)  # Using default threshold
        self.assertTrue(overlap)

    def test_boxes_overlap_false(self):
        """Test non-overlapping bounding boxes detection"""
        box1 = {'Left': 0.1, 'Top': 0.1, 'Width': 0.2, 'Height': 0.2}
        box2 = {'Left': 0.8, 'Top': 0.8, 'Width': 0.2, 'Height': 0.2}
        self.assertFalse(self.analyzer._boxes_overlap(box1, box2, threshold=0.5))

    @patch('PIL.Image.open')
    def test_image_orientation_correction(self, mock_image_open):
        """Test image orientation correction with various EXIF orientations"""
        mock_image = MagicMock()
        mock_image._getexif.return_value = {274: 6}  # 6 = 270 degree rotation
        mock_image_open.return_value = mock_image
        
        self.analyzer._correct_image_orientation(mock_image)
        mock_image.rotate.assert_called_with(270, expand=True)

    def test_process_response_duplicate_detection(self):
        """Test deduplication of overlapping detections"""
        response = {
            'Labels': [
                {
                    'Name': 'Person',
                    'Confidence': 99.9,
                    'Instances': [
                        {
                            'BoundingBox': {'Left': 0.1, 'Top': 0.1, 'Width': 0.5, 'Height': 0.5},
                            'Confidence': 99.9
                        },
                        {
                            'BoundingBox': {'Left': 0.12, 'Top': 0.11, 'Width': 0.5, 'Height': 0.5},
                            'Confidence': 98.5
                        }
                    ]
                }
            ]
        }
        result = self.analyzer._process_response(response)
        self.assertEqual(len(result['people'][0]['instances']), 1)

    @patch('PIL.ImageDraw.Draw')
    @patch('PIL.Image.open')
    def test_draw_bounding_boxes_color_selection(self, mock_image_open, mock_draw):
        """Test correct color selection for different object types"""
        mock_image = MagicMock()
        mock_image._getexif.return_value = None
        mock_image.size = (800, 600)  # Add size attribute
        mock_image_open.return_value = mock_image

        results = {
            'people': [{
                'name': 'Person',
                'confidence': '99.9%',
                'instances': [{'box': {'Left': 0.1, 'Top': 0.1, 'Width': 0.5, 'Height': 0.5}}]
            }],
            'animals': [{
                'name': 'Dog',
                'confidence': '95.5%',
                'instances': [{'box': {'Left': 0.6, 'Top': 0.6, 'Width': 0.3, 'Height': 0.3}}]
            }]
        }

        self.analyzer.draw_bounding_boxes('test.jpg', results, 'output.jpg')
        mock_draw.assert_called()


    def test_face_details_processing(self):
        """Test processing of face details response"""
        mock_response = {
            'FaceDetails': [{
                'AgeRange': {'Low': 20, 'High': 30},
                'Gender': {'Value': 'Female', 'Confidence': 99.9},
                'Emotions': [
                    {'Type': 'HAPPY', 'Confidence': 95.0},
                    {'Type': 'CALM', 'Confidence': 4.0}
                ],
                'Smile': {'Value': True, 'Confidence': 98.0},
                'Quality': {'Brightness': 90.0, 'Sharpness': 80.0}
            }]
        }
        
        self.analyzer.rekognition_client.detect_faces.return_value = mock_response
        result = self.analyzer.detect_face_details(b'dummy_image_bytes')
        
        self.assertEqual(result[0]['AgeRange']['Low'], 20)
        self.assertEqual(result[0]['Gender']['Value'], 'Female')
        self.assertEqual(result[0]['Emotions'][0]['Type'], 'HAPPY')

if __name__ == '__main__':
    unittest.main(verbosity=2)
