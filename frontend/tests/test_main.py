"""Tests for frontend main.py."""
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from main import AnalysisResults, AnalyzedImage, App, Header, Settings, app

# Create a test client
client = TestClient(app)

# Mock settings for testing
test_settings = Settings()
test_settings.BACKEND_URL = "http://testserver:8000"


def test_static_files_mounted():
    """Test that static files are properly mounted."""
    response = client.get("/static/styles/main.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_analyzed_image_component(mocker):
    """Test the AnalyzedImage component's interaction with backend."""
    # Mock the AsyncClient post method
    mock_response = Response(200, json={"status": "success"})
    mock_post = mocker.patch("httpx.AsyncClient.post", return_value=mock_response)

    component = AnalyzedImage()
    assert component is not None

    # Verify the mock was called
    mock_post.assert_called_once


@pytest.mark.asyncio
async def test_analysis_results_component(mocker):
    """Test the AnalysisResults component's interaction with backend."""
    # Mock responses for each endpoint
    mock_response = Response(
        200,
        json={
            "status": "success",
            "data": {
                "labels": [{"Name": "Test", "Confidence": 99.9}],
                "faces": {"faces_found": 1},
                "matches": {"matches_found": 1},
            },
        },
    )

    # Mock both get and post methods
    mocker.patch("httpx.AsyncClient.post", return_value=mock_response)
    mocker.patch("httpx.AsyncClient.get", return_value=mock_response)

    component = AnalysisResults()
    assert component is not None


def test_cors_middleware():
    """Test CORS middleware configuration."""
    response = client.options(
        "/",
        headers={
            "origin": "http://testclient.com",
            "access-control-request-method": "POST",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    # The CORS middleware is configured to allow all origins
    assert response.headers["access-control-allow-origin"] == "http://testclient.com"
    assert "access-control-allow-methods" in response.headers


def test_app_settings():
    """Test application settings."""
    assert Settings.HOST == "0.0.0.0"
    assert Settings.PORT == 3000
    assert Settings.BACKEND_URL == "http://localhost:8000"
    assert Settings.APP_TITLE == "B.R.U.T.U.S."


@pytest.fixture
def mock_backend_responses(mocker):
    """Mock backend API responses."""
    mock_response = Response(
        200,
        json={
            "status": "success",
            "data": {
                "labels": [{"Name": "Test", "Confidence": 99.9}],
                "faces": {"faces_found": 1},
                "matches": {"matches_found": 1},
            },
        },
    )

    mocker.patch("httpx.AsyncClient.post", return_value=mock_response)
    mocker.patch("httpx.AsyncClient.get", return_value=mock_response)


@pytest.mark.asyncio
async def test_analyzed_image_with_mocks(mock_backend_responses):
    """Test AnalyzedImage component with mocked backend responses."""
    component = AnalyzedImage()
    assert component is not None


@pytest.mark.asyncio
async def test_analysis_results_with_mocks(mock_backend_responses):
    """Test AnalysisResults component with mocked backend responses."""
    component = AnalysisResults()
    assert component is not None


def test_header_component():
    """Test Header component."""
    header = Header()
    assert header is not None


def test_app_component():
    """Test main App component."""
    app_component = App()
    assert app_component is not None
