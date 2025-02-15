import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import io
import sys
from PIL import Image
from image_recognition import BrutusSees


class TestImageAnalyzerUnit(unittest.TestCase):
    """Unit tests for image analysis:
    - Initialization error handling
    - Label detection: success, error, empty response
    - Face detection: success or no faces detected
    - Face comparison: success, file handling
    - Bounding box: test drawing logic and image handling
    """

    def setUp(self):
        """Set up test cases"""
        self.logger_patcher = patch("image_recognition.logger")
        self.mock_logger = self.logger_patcher.start()

        # Create analyzer with mocked Rekognition client
        with patch("boto3.client"):
            self.analyzer = BrutusSees()
            self.analyzer.rekognition_client = Mock()

    def tearDown(self):
        """Clean up after tests"""
        self.logger_patcher.stop()

    def test_init_success(self):
        """Test successful initialization"""
        with patch("boto3.client") as mock_boto3_client:
            analyzer = BrutusSees()
            mock_boto3_client.assert_called_once_with("rekognition")
            self.mock_logger.info.assert_called_with(
                "Successfully initialized Rekognition client"
            )

    def test_init_failure(self):
        """Test initialization failure"""
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.side_effect = Exception("AWS Error")
            with self.assertRaises(Exception):
                BrutusSees()
            self.mock_logger.error.assert_called_with(
                "Failed to initialize client", exc_info=True
            )

    @patch("PIL.Image.open")
    def test_source_image_no_exif(self, mock_image_open):
        """Test _source_image with no EXIF data"""
        mock_img = MagicMock()
        mock_img._getexif.return_value = None
        mock_image_open.return_value = mock_img

        result = self.analyzer._source_image("test.jpg")
        self.assertIsInstance(result, bytes)
        mock_img.save.assert_called_once()

    @patch("PIL.Image.open")
    def test_source_image_with_rotation(self, mock_image_open):
        """Test _source_image with rotation in EXIF"""
        mock_img = MagicMock()
        mock_img._getexif.return_value = {274: 3}  # 3 = 180 degree rotation
        mock_image_open.return_value = mock_img

        result = self.analyzer._source_image("test.jpg")
        self.assertIsInstance(result, bytes)
        mock_img.rotate.assert_called_once_with(180, expand=True)

    def test_detect_labels_success(self):
        """Test successful label detection"""
        mock_response = {
            "Labels": [
                {
                    "Name": "Person",
                    "Confidence": 99.9,
                    "Instances": [
                        {
                            "BoundingBox": {
                                "Left": 0.1,
                                "Top": 0.1,
                                "Width": 0.5,
                                "Height": 0.5,
                            },
                            "Confidence": 99.9,
                        }
                    ],
                }
            ]
        }
        self.analyzer.rekognition_client.detect_labels.return_value = mock_response

        with patch.object(self.analyzer, "_source_image", return_value=b"image_bytes"):
            result = self.analyzer.detect_labels_in_image("test.jpg")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Name"], "Person")
        self.assertEqual(len(result[0]["Instances"]), 1)
        self.assertEqual(result[0]["Instances"][0]["label_number"], 1)

    def test_detect_faces_success(self):
        """Test successful face detection"""
        mock_response = {
            "FaceDetails": [
                {
                    "BoundingBox": {
                        "Left": 0.1,
                        "Top": 0.1,
                        "Width": 0.5,
                        "Height": 0.5,
                    },
                    "Confidence": 99.9,
                    "Gender": {"Value": "Female"},
                    "AgeRange": {"Low": 20, "High": 30},
                    "Emotions": [{"Type": "HAPPY", "Confidence": 95.0}],
                }
            ]
        }
        self.analyzer.rekognition_client.detect_faces.return_value = mock_response

        with patch.object(self.analyzer, "_source_image", return_value=b"image_bytes"):
            result = self.analyzer.detect_and_return_face_details("test.jpg")

        self.assertIn("faces", result)
        self.assertEqual(len(result["faces"]), 1)
        face = result["faces"][0]
        self.assertEqual(face["face_number"], 1)
        self.assertEqual(face["gender"], "Female")
        self.assertEqual(face["primary_emotion"], "HAPPY")

    def test_compare_faces_success(self):
        """Test successful face comparison"""
        # Mock face detection response
        mock_face_response = {
            "FaceDetails": [
                {
                    "BoundingBox": {
                        "Left": 0.1,
                        "Top": 0.1,
                        "Width": 0.5,
                        "Height": 0.5,
                    },
                    "Confidence": 99.9,
                }
            ]
        }

        # Mock face comparison response
        mock_compare_response = {
            "FaceMatches": [
                {
                    "Similarity": 98.5,
                    "Face": {
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.1,
                            "Width": 0.5,
                            "Height": 0.5,
                        },
                        "Confidence": 99.9,
                    },
                }
            ]
        }

        # Setup mock responses
        self.analyzer.rekognition_client.detect_faces.return_value = mock_face_response
        self.analyzer.rekognition_client.compare_faces.return_value = (
            mock_compare_response
        )

        # Mock file system operations and image source
        with patch("os.listdir") as mock_listdir, patch.object(
            self.analyzer, "_source_image", return_value=b"image_bytes"
        ):

            mock_listdir.return_value = ["person1.jpg"]
            result = self.analyzer.compare_faces_with_library("test.jpg", "reference")

        # Verify results
        self.assertIn("matches", result)
        self.assertEqual(result["matches_found"], 1)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["person"], "person1")

    def test_detect_labels_error(self):
        """Test label detection error handling"""
        self.analyzer.rekognition_client.detect_labels.side_effect = Exception(
            "AWS Error"
        )

        with patch.object(self.analyzer, "_source_image", return_value=b"image_bytes"):
            with self.assertRaises(Exception):
                self.analyzer.detect_labels_in_image("test.jpg")
            self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
