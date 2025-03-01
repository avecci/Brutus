"""Main web app for Brutus frontend. Uses ReactPY with FastAPI as backend."""
import os
import time
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from reactpy import component, html, use_effect, use_state
from reactpy.backend.fastapi import configure

from logging_utils import setup_logger

logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application configuration settings."""

    HOST: str = "0.0.0.0"
    PORT: int = 3000
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    APP_TITLE: str = "B.R.U.T.U.S."


settings = Settings()

app = FastAPI(title=settings.APP_TITLE)

logger.info(
    "Starting frontend application",
    extra={
        "host": settings.HOST,
        "port": settings.PORT,
        "backend_url": settings.BACKEND_URL,
    },
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
logger.info(f"Mounted static files directory: {static_dir}")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("Configured CORS middleware")


@component
def Header():
    """Header component displaying the application title."""
    return html.header({"class": "header"}, html.h1(settings.APP_TITLE))


@component
def AnalyzedImage():
    """Component to display the analyzed image."""
    timestamp, set_timestamp = use_state(0)  # Add timestamp state
    loading, set_loading = use_state(True)
    error, set_error = use_state(None)

    async def process_image():
        try:
            logger.info("Starting image processing")
            set_loading(True)
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug("Calling save image endpoint")
                save_response = await client.post(f"{settings.BACKEND_URL}/image/save")
                if save_response.status_code != 200:
                    error_msg = "Failed to process image"
                    logger.error(f"{error_msg}: {save_response.status_code}")
                    set_error(error_msg)
                    return

                set_timestamp(
                    int(time.time())
                )  # Update timestamp to force image refresh
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing image: {error_msg}", exc_info=True)
            set_error(error_msg)
        finally:
            set_loading(False)
            logger.info("Image processing completed")

    use_effect(process_image)

    if loading:
        logger.debug("Rendering loading state")
        return html.div(
            {"class": "loading"},
            [html.div({"class": "spinner"}), html.p("Processing image...")],
        )

    if error:
        logger.debug(f"Rendering error state: {error}")
        return html.div({"class": "error"}, f"Error: {error}")

    # Update image URL to use new endpoint
    image_url = f"{settings.BACKEND_URL}/image/latest?t={timestamp}"
    logger.debug(f"Rendering analyzed image with URL: {image_url}")

    return html.div(
        {"class": "image-container"},
        [
            html.h2("Analyzed Image"),
            html.img(
                {
                    "src": image_url,
                    "alt": "Analyzed Image",
                    "class": "analyzed-image",
                    "style": {
                        "max-width": "800px",
                        "width": "100%",
                        "height": "auto",
                        "object-fit": "contain",
                    },
                }
            ),
        ],
    )


@component
def AnalysisResults():
    """Component to display analysis results from the backend."""
    results, set_results = use_state({})
    loading, set_loading = use_state(True)
    error, set_error = use_state(None)
    analyzed, set_analyzed = use_state(False)

    async def fetch_analysis_results():
        logger.info("Starting analysis results fetch")
        try:
            set_loading(True)
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug("Calling save image endpoint")
                save_response = await client.post(f"{settings.BACKEND_URL}/image/save")
                if save_response.status_code != 200:
                    error_msg = "Failed to process image"
                    logger.error(f"{error_msg}: {save_response.status_code}")
                    set_error(error_msg)
                    return

                # Then fetch analysis results
                logger.debug("Fetching analysis results")
                labels_response = await client.get(
                    f"{settings.BACKEND_URL}/image/analyze"
                )
                faces_response = await client.get(
                    f"{settings.BACKEND_URL}/image/analyze/faces"
                )
                facial_recognition = await client.get(
                    f"{settings.BACKEND_URL}/image/analyze/facial-recognition"
                )

                if all(
                    r.status_code == 200
                    for r in [labels_response, faces_response, facial_recognition]
                ):
                    labels_data = (
                        labels_response.json().get("data", {}).get("labels", [])
                    )
                    faces_data = faces_response.json().get("data", {})

                    logger.info(
                        "Analysis results fetched successfully",
                        extra={
                            "labels_count": len(labels_data),
                            "faces_found": faces_data.get("faces", {}).get(
                                "faces_found", 0
                            ),
                        },
                    )

                    set_results(
                        {
                            "labels": labels_data,
                            "faces": faces_data,
                            "facial_recognition": facial_recognition.json(),
                        }
                    )

                    set_analyzed(True)
                else:
                    error_msg = "Failed to fetch analysis results"
                    logger.error(f"{error_msg}: Response status codes not all 200")
                    set_error(error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching analysis results: {error_msg}", exc_info=True)
            set_error(error_msg)
        finally:
            set_loading(False)
            logger.info("Analysis results fetch completed")

    use_effect(fetch_analysis_results, [analyzed])

    if loading:
        return html.div(
            {"class": "loading"},
            [html.div({"class": "spinner"}), html.p("Loading analysis results...")],
        )

    if error:
        return html.div({"class": "error"}, f"Error: {error}")

    def format_label(label):
        """Format label information."""
        name = label.get("Name", "Unknown")
        confidence = label.get("Confidence", 0)
        return f"{name} ({confidence:.2f}%)"

    def format_face_details(face):
        """Format face details."""
        details = []

        # Age Range
        age_range = face.get("age_range", {})
        details.append(
            f"Age Range: {age_range.get('Low', 'N/A')}-{age_range.get('High', 'N/A')}"
        )

        # Gender
        details.append(f"Gender: {face.get('gender', 'Unknown')}")

        # Emotion
        details.append(
            f"Emotion: {face.get('primary_emotion', 'Unknown')} "
            f"({face.get('emotion_confidence', 0):.2f}%)"
        )

        # Characteristics
        chars = face.get("characteristics", {})
        if chars:
            details.append("Characteristics:")
            for char_name, char_data in chars.items():
                if isinstance(char_data, dict):
                    details.append(
                        f"  - {char_name}: {char_data.get('value', 'N/A')} "
                        f"({char_data.get('confidence', 0):.2f}%)"
                    )

        return html.div([html.p(detail) for detail in details])

    return html.div(
        {"class": "analysis-results"},
        [
            # Labels Section
            html.div(
                {"class": "result-section"},
                [
                    html.h3("Detected Labels"),
                    html.ul(
                        [
                            html.li(format_label(label))
                            for label in results.get("labels", [])
                        ]
                    )
                    if results.get("labels")
                    else html.p("No labels detected"),
                ],
            ),
            # Face Details Section
            html.div(
                {"class": "result-section"},
                [
                    html.h3("Face Details"),
                    html.p(
                        f"Number of faces found: {results.get('faces', {}).get('faces', {}).get('faces_found', 0)}"
                    ),
                    html.div(
                        {"class": "faces-list"},
                        [
                            html.div(
                                {"class": "face-detail"},
                                [
                                    html.h4(f"Face #{face.get('face_number', 'N/A')}"),
                                    format_face_details(face),
                                ],
                            )
                            for face in results.get("faces", {})
                            .get("faces", {})
                            .get("faces", [])
                        ],
                    )
                    if results.get("faces", {}).get("faces", {}).get("faces")
                    else html.p("No face details available"),
                ],
            ),
            # Facial Recognition Section
            html.div(
                {"class": "result-section"},
                [
                    html.h3("Recognized Faces"),
                    html.p(
                        f"Number of matches found: {results.get('facial_recognition', {}).get('data', {}).get('matches', {}).get('matches_found', 0)}"
                    ),
                    html.ul(
                        [
                            html.li(
                                f"{match.get('person', 'Unknown')} (Similarity: {match.get('similarity', 0):.2f}%, Confidence: {match.get('confidence', 0):.2f}%)"
                            )
                            for match in results.get("facial_recognition", {})
                            .get("data", {})
                            .get("matches", {})
                            .get("matches", [])
                        ]
                    )
                    if results.get("facial_recognition", {})
                    .get("data", {})
                    .get("matches", {})
                    .get("matches")
                    else html.p("No faces recognized"),
                ],
            ),
        ],
    )


@component
def App():
    """Root component of the application."""
    return html.div(
        {"class": "container"},
        [
            html.title("B.R.U.T.U.S."),
            html.link({"rel": "stylesheet", "href": "/static/styles/main.css"}),
            Header(),
            AnalyzedImage(),
            AnalysisResults(),
        ],
    )


# Configure the app with the root component
configure(app, App)
logger.info("Configured ReactPY with FastAPI application")


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
