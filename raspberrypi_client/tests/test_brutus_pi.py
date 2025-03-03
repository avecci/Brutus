"""Connectivity tests for terminal client."""
from unittest.mock import Mock, mock_open, patch

import pygame
import pytest
from brutus_pi import AudioHandler, BackendHandler, PictureHandler, Settings


@pytest.fixture
def settings():
    """Fixture to provide a Settings instance for testing."""
    return Settings()


@pytest.fixture
def mock_audio_handler():
    """Fixture to mock pygame.mixer for testing."""
    with patch("pygame.mixer.init"), patch("pygame.mixer.music"):
        handler = AudioHandler()
        yield handler


@pytest.fixture
def mock_picture_handler():
    """Fixture to mock PictureHandler for testing."""
    with patch("os.makedirs"):
        handler = PictureHandler()
        yield handler


class TestPictureHandler:
    """Tests for the PictureHandler class."""

    def test_capture_rotate_save_success(self, tmp_path):
        """Test successful image capture and rotation."""
        with patch("os.system", return_value=0), patch("PIL.Image.open") as mock_image:
            # Setup mock image
            mock_img = Mock()
            mock_img.rotate.return_value = mock_img
            mock_image.return_value.__enter__.return_value = mock_img

            handler = PictureHandler()
            test_path = str(tmp_path / "test.jpg")
            handler.capture_rotate_and_save(test_path)

            # Verify rotation was called
            mock_img.rotate.assert_called_once_with(-90, expand=True)

    def test_capture_failure(self):
        """Test handling of rpicam-still failure."""
        with patch("os.system", return_value=1):
            handler = PictureHandler()
            with pytest.raises(RuntimeError, match="rpicam-still failed"):
                handler.capture_rotate_and_save("test.jpg")

    def test_upload_image_success(self, requests_mock):
        """Test successful image upload."""
        handler = PictureHandler()
        requests_mock.post(
            handler.BACKEND_API_IMAGE_UPLOAD_URL, json={"status": "success"}
        )

        with patch("builtins.open", mock_open(read_data=b"test")):
            response = handler.upload_image("test.jpg")
            assert response == {"status": "success"}


class TestAudioHandler:
    """Tests for the AudioHandler class."""

    def test_play_audio_success(self, mock_audio_handler):
        """Test successful audio playback."""
        with patch("pygame.mixer.music.get_busy", side_effect=[True, False]):
            mock_audio_handler.play_audio("test.mp3")
            pygame.mixer.music.load.assert_called_once_with("test.mp3")
            pygame.mixer.music.play.assert_called_once()

    def test_cleanup(self, mock_audio_handler):
        """Test audio handler cleanup."""
        mock_audio_handler.cleanup()
        pygame.mixer.quit.assert_called_once()


class TestBackendHandler:
    """Tests for the BackendHandler class."""

    def test_health_check_success(self, requests_mock):
        """Test successful backend health check."""
        requests_mock.get(f"{Settings.BACKEND_URL}/health", json={"status": "Healthy"})
        assert BackendHandler.check_backend_health() is True

    def test_health_check_failure(self, requests_mock):
        """Test failed backend health check."""
        requests_mock.get(f"{Settings.BACKEND_URL}/health", status_code=500)
        assert BackendHandler.check_backend_health() is False
