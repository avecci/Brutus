"""Test image recognition class methods individually."""
import unittest
from unittest.mock import MagicMock, Mock, patch

from botocore.exceptions import ClientError

from image_recognition import BrutusEyes


class TestImageAnalyzerUnit(unittest.TestCase):
    """Unit tests for image analysis.

    - Initialization error handling
    - Label detection: success, error, empty response
    - Face detection: success or no faces detected
    - Face comparison: success, file handling
    - Bounding box: test drawing logic and image handling
    """

    def setUp(self):
        """Set up test cases."""
        self.logger_patcher = patch("image_recognition.logger")
        self.mock_logger = self.logger_patcher.start()

        # Create analyzer with mocked Rekognition client
        with patch("boto3.client"):
            self.analyzer = BrutusEyes()
            self.analyzer.rekognition_client = Mock()

    def tearDown(self):
        """Clean up after tests."""
        self.logger_patcher.stop()

    def test_init_success(self):
        """Test successful initialization."""
        mock_boto3_client = Mock()
        with patch("boto3.client", return_value=mock_boto3_client) as mock_client:
            BrutusEyes()
            mock_client.assert_called_once_with("rekognition")

    def test_init_failure(self):
        """Test initialization failure."""
        with patch("boto3.client") as mock_boto3_client:
            error_response = {
                "Error": {
                    "Code": "InvalidCredentials",
                    "Message": "Invalid AWS credentials",
                }
            }
            mock_boto3_client.side_effect = ClientError(error_response, "CreateClient")

            with self.assertRaises(ClientError) as context:
                BrutusEyes()

            # Verify the specific error
            self.assertEqual(
                context.exception.response["Error"]["Code"], "InvalidCredentials"
            )
            self.mock_logger.error.assert_called_with(
                "Failed to initialize client", exc_info=True
            )

    @patch("PIL.Image.open")
    def test_source_image_no_exif(self, mock_image_open):
        """Test _source_image with no EXIF data."""
        mock_img = MagicMock()
        mock_img.getexif.return_value = None
        mock_image_open.return_value = mock_img

        result = self.analyzer._source_image("test.jpg")
        self.assertIsInstance(result, bytes)
        mock_img.save.assert_called_once()

    @patch("PIL.Image.open")
    def test_source_image_with_rotation(self, mock_image_open):
        """Test _source_image with rotation in EXIF."""
        # Create a more complete mock image
        mock_img = MagicMock()
        mock_img.getexif.return_value = {274: 3}  # 3 = 180 degree rotation
        mock_img.rotate.return_value = mock_img  # Return the mock for method chaining
        mock_image_open.return_value = mock_img

        # Create a BytesIO mock that returns actual bytes
        mock_bytes_io = MagicMock()
        mock_bytes_io.getvalue.return_value = b"test_image_bytes"

        with patch("io.BytesIO") as mock_bytesio_class:
            # Configure the context manager behavior
            mock_bytesio_instance = MagicMock()
            mock_bytesio_instance.__enter__.return_value = mock_bytes_io
            mock_bytesio_class.return_value = mock_bytesio_instance

            result = self.analyzer._source_image("test.jpg")

            # Verify the results
            self.assertIsInstance(result, bytes)
            self.assertEqual(result, b"test_image_bytes")
            mock_img.rotate.assert_called_once_with(180, expand=True)
            mock_img.save.assert_called_once()
            mock_bytes_io.getvalue.assert_called_once()

    def test_detect_labels_success(self):
        """Test successful label detection."""
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
        """Test successful face detection."""
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
        """Test successful face comparison."""
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
        """Test label detection error handling."""
        # Mock a specific AWS exception
        error_response = {
            "Error": {
                "Code": "InvalidImageFormatException",
                "Message": "Invalid image format",
            }
        }
        self.analyzer.rekognition_client.detect_labels.side_effect = ClientError(
            error_response, "DetectLabels"
        )

        with patch.object(self.analyzer, "_source_image", return_value=b"image_bytes"):
            with self.assertRaises(ClientError) as context:
                self.analyzer.detect_labels_in_image("test.jpg")

            # Verify the error details
            self.assertEqual(
                context.exception.response["Error"]["Code"],
                "InvalidImageFormatException",
            )
            self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
