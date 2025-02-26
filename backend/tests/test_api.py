"""Tests for FastAPI endpoints."""
import io
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app

client = TestClient(app)


@pytest.fixture
def mock_brutus_eyes():
    """Create a mock BrutusEyes instance."""
    with patch("api.brutus_eyes") as mock:
        yield mock


@pytest.fixture
def test_image():
    """Create a temporary test image."""
    with Image.new("RGB", (100, 100), color="red") as img:
        try:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)
            return img_byte_arr
        except (IOError, ValueError) as e:
            pytest.fail(f"Failed to create test image: {str(e)}")


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Brutus backend API"
    assert response.json()["status"] == "online"
    assert "version" in response.json()
    assert "endpoints" in response.json()
    assert "description" in response.json()


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200


def test_analyze_image_success(mock_brutus_eyes):
    """Test successful image analysis."""
    mock_brutus_eyes.detect_labels_in_image.return_value = [
        {"Name": "Person", "Confidence": 99.9}
    ]

    response = client.get("/analyze/image")

    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"
    assert "data" in response.json()
    mock_brutus_eyes.detect_labels_in_image.assert_called_once()


def test_analyze_image_not_found():
    """Test image analysis with non-existent image."""
    response = client.get("/analyze/image?input_path=nonexistent.jpg")
    assert response.status_code == 404


def test_analyze_faces_success(mock_brutus_eyes):
    """Test successful face detection."""
    mock_brutus_eyes.detect_and_return_face_details.return_value = [
        {"BoundingBox": {"Width": 0.5, "Height": 0.5, "Left": 0.25, "Top": 0.25}}
    ]

    response = client.get("/analyze/faces")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "faces" in response.json()["data"]
    mock_brutus_eyes.detect_and_return_face_details.assert_called_once()


def test_analyze_faces_not_found():
    """Test face detection with non-existent image."""
    response = client.get("/analyze/faces?input_path=nonexistent.jpg")
    assert response.status_code == 404


def test_facial_recognition_success(mock_brutus_eyes):
    """Test successful facial recognition."""
    mock_brutus_eyes.compare_faces_with_library.return_value = {
        "matches_found": 1,
        "matches": [{"Name": "Test Person", "Similarity": 99.9}],
    }

    response = client.get("/analyze/facial-recognition")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "matches" in response.json()["data"]
    mock_brutus_eyes.compare_faces_with_library.assert_called_once()


def test_facial_recognition_input_not_found():
    """Test facial recognition with non-existent input image."""
    response = client.get("/analyze/facial-recognition?input_path=nonexistent.jpg")
    assert response.status_code == 404


def test_facial_recognition_reference_not_found():
    """Test facial recognition with non-existent reference directory."""
    response = client.get("/analyze/facial-recognition?reference_dir=nonexistent")
    assert response.status_code == 404


def test_analyze_all_success(mock_brutus_eyes):
    """Test successful complete analysis."""
    mock_brutus_eyes.detect_labels_in_image.return_value = [
        {"Name": "Person", "Confidence": 99.9}
    ]
    mock_brutus_eyes.detect_and_return_face_details.return_value = [{"BoundingBox": {}}]
    mock_brutus_eyes.compare_faces_with_library.return_value = {"matches_found": 1}

    response = client.get("/analyze/all")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert all(
        k in response.json()["data"] for k in ["labels", "faces", "face_matches"]
    )
    mock_brutus_eyes.detect_labels_in_image.assert_called_once()
    mock_brutus_eyes.detect_and_return_face_details.assert_called_once()
    mock_brutus_eyes.compare_faces_with_library.assert_called_once()


def test_analyze_all_input_not_found():
    """Test complete analysis with non-existent input image."""
    response = client.get("/analyze/all?input_path=nonexistent.jpg")
    assert response.status_code == 404


def test_analyze_all_known_faces_not_found():
    """Test complete analysis with non-existent known faces directory."""
    response = client.get("/analyze/all?known_faces_dir=nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_save_analyzed_image_success(mock_brutus_eyes, test_image):
    """Test successful image saving with analysis."""
    test_output = "test_output.jpg"
    mock_brutus_eyes.draw_bounding_boxes.return_value = Image.new("RGB", (100, 100))

    with patch("pathlib.Path.exists", return_value=True):
        response = client.post(f"/analyze/save-image?output_path={test_output}")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "output_image" in response.json()["data"]
    mock_brutus_eyes.draw_bounding_boxes.assert_called_once()

    # Cleanup
    if os.path.exists(test_output):
        os.remove(test_output)


def test_save_analyzed_image_input_not_found():
    """Test image saving with non-existent input image."""
    response = client.post("/analyze/save-image?input_path=nonexistent.jpg")
    assert response.status_code == 404  # API returns 500 for file not found


def test_save_analyzed_image_known_faces_not_found():
    """Test image saving with non-existent known faces directory."""
    response = client.post("/analyze/save-image?known_faces_dir=nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_image_success(test_image):
    """Test successful image upload."""
    files = {"file": ("test.jpg", test_image, "image/jpeg")}
    response = client.post("/upload/image", files=files)

    assert response.status_code == 201
    assert response.json()["status"] == "success"
    assert "filename" in response.json()["data"]


def test_upload_image_no_file():
    """Test image upload with no file."""
    response = client.post("/upload/image")
    assert response.status_code == 422


def test_upload_image_invalid_type():
    """Test image upload with invalid file type."""
    files = {"file": ("test.txt", b"test content", "text/plain")}
    response = client.post("/upload/image", files=files)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_generate_speech_success():
    """Test successful speech generation."""
    test_text = "Hello, World!"

    # Mock the speech generator
    with patch("api.brutus_voice") as mock_voice:
        # Configure mock to write some test audio data
        def mock_text_to_speech(text, output_path):
            with open(output_path, "wb") as f:
                f.write(b"test audio data")
            return True

        mock_voice.text_to_speech.side_effect = mock_text_to_speech

        # Make request
        response = client.post(f"/generate/speech?text={test_text}")

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="speech.mp3"'
        )
        assert response.content == b"test audio data"

        # Verify mock was called correctly
        mock_voice.text_to_speech.assert_called_once()
        call_args = mock_voice.text_to_speech.call_args[0]
        assert call_args[0] == test_text
        assert call_args[1].endswith(".mp3")


@pytest.mark.asyncio
async def test_generate_speech_failure():
    """Test speech generation failure."""
    with patch("api.brutus_voice") as mock_voice:
        mock_voice.text_to_speech.return_value = None  # Simulate failure

        response = client.post("/generate/speech?text=test")

        assert response.status_code == 500
        assert "Speech generation failed" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main(["-v"])
