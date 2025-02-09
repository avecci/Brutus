import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, ANY, call
from pathlib import Path
from image_recognition import ImageAnalyzer

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


class TestImageAnalyzerUnit(unittest.TestCase):
    """Unit tests for ImageAnalyzer class"""

    def setUp(self):
        """Set up test cases"""
        # Configure logging to avoid writing to actual log files during tests
        self.logger_patcher = patch("image_recognition.logger")
        self.mock_logger = self.logger_patcher.start()

        # Create analyzer with mocked Rekognition client
        self.analyzer = ImageAnalyzer()
        self.analyzer.rekognition_client = Mock()

    def assertLogMessageContains(self, level, expected_message):
        """Helper method to check if a log message exists"""
        log_method = getattr(self.mock_logger, level)
        calls = log_method.call_args_list
        self.assertTrue(
            any(
                call[0][0] == expected_message
                and isinstance(call[1].get("extra", {}).get("request_id"), str)
                for call in calls
            ),
            f"Expected log message '{expected_message}' not found",
        )

    def tearDown(self):
        """Clean up after tests"""
        self.logger_patcher.stop()

    def test_init_with_custom_client(self):
        """Test initialization with custom rekognition client"""
        mock_client = Mock()
        self.analyzer.rekognition_client = mock_client
        self.assertEqual(self.analyzer.rekognition_client, mock_client)
        self.mock_logger.info.assert_called()

    @patch("boto3.client")
    def test_init_aws_error(self, mock_boto3_client):
        """Test handling of AWS initialization error"""
        mock_boto3_client.side_effect = Exception("AWS Error")
        with self.assertRaises(Exception) as context:
            ImageAnalyzer()
        self.assertIn("AWS Error", str(context.exception))
        self.mock_logger.error.assert_called()

    def test_process_response_empty(self):
        """Test processing of empty response"""
        empty_response = {"Labels": []}
        result = self.analyzer._process_response(empty_response)

        # Check empty lists for all categories
        self.assertEqual(result["people"], [])
        self.assertEqual(result["animals"], [])
        self.assertEqual(result["objects"], [])
        # Check image properties exists and is empty
        self.assertEqual(result["image_properties"], {})

        # Verify logging occurred
        self.mock_logger.debug.assert_called()

    def test_process_response_error_handling(self):
        """Test error handling in response processing"""
        invalid_response = {"InvalidKey": []}

        with self.assertRaises(Exception):
            self.analyzer._process_response(invalid_response)

        # Verify error was logged with new structured format
        self.mock_logger.error.assert_called_with(
            "Error processing Rekognition response",
            extra={
                "request_id": ANY,
                "error": {"type": ANY, "message": ANY, "response_keys": ["InvalidKey"]},
            },
            exc_info=True,
        )

    @patch("PIL.Image.open")
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
            "Labels": [
                {
                    "Name": "Person",
                    "Confidence": 99.9,
                    "Instances": [
                        {
                            "BoundingBox": {
                                "Left": 0.1,
                                "Top": 0.2,
                                "Width": 0.3,
                                "Height": 0.4,
                            },
                            "Confidence": 99.9,
                        },
                        {
                            "BoundingBox": {
                                "Left": 0.11,
                                "Top": 0.21,
                                "Width": 0.3,
                                "Height": 0.4,
                            },
                            "Confidence": 98.5,
                        },
                    ],
                }
            ]
        }

        result = self.analyzer._process_response(response)
        # Check that we only have one person detection
        self.assertEqual(len(result["people"]), 1)
        # Check that the bounding box exists
        self.assertIn("bounding_box", result["people"][0])
        # Check that we kept the highest confidence detection
        self.assertAlmostEqual(result["people"][0]["confidence"], 99.9)

    @patch("PIL.ImageDraw.Draw")
    @patch("PIL.Image.open")
    def test_draw_bounding_boxes_color_selection(self, mock_image_open, mock_draw):
        """Test correct color selection for different object types"""
        mock_image = MagicMock()
        mock_image._getexif.return_value = None
        mock_image.size = (800, 600)
        mock_image_open.return_value = mock_image

        results = {
            "people": [
                {
                    "name": "Person",
                    "confidence": 99.9,
                    "bounding_box": {
                        "Left": 0.1,
                        "Top": 0.1,
                        "Width": 0.5,
                        "Height": 0.5,
                    },
                }
            ],
            "animals": [
                {
                    "name": "Dog",
                    "confidence": 95.5,
                    "bounding_box": {
                        "Left": 0.6,
                        "Top": 0.6,
                        "Width": 0.3,
                        "Height": 0.3,
                    },
                }
            ],
        }

        self.analyzer.draw_bounding_boxes("test.jpg", results, "output.jpg")
        mock_draw.assert_called()

        # Check that logging occurred
        self.mock_logger.debug.assert_called()


def test_face_details_processing(self):
    """Test processing of face details response"""
    mock_response = {
        "FaceDetails": [
            {
                "AgeRange": {"Low": 20, "High": 30},
                "Gender": {"Value": "Female", "Confidence": 99.9},
                "Emotions": [
                    {"Type": "HAPPY", "Confidence": 95.0},
                    {"Type": "CALM", "Confidence": 4.0},
                ],
                "Smile": {"Value": True, "Confidence": 98.0},
                "Quality": {"Brightness": 90.0, "Sharpness": 80.0},
            }
        ]
    }

    self.analyzer.rekognition_client.detect_faces.return_value = mock_response
    result = self.analyzer.detect_face_details(b"dummy_image_bytes")

    # Verify the response structure
    self.assertEqual(result[0]["AgeRange"]["Low"], 20)
    self.assertEqual(result[0]["Gender"]["Value"], "Female")
    self.assertEqual(result[0]["Emotions"][0]["Type"], "HAPPY")

    # Verify logging
    self.mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
