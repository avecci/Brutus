import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from reactpy.backend.fastapi import configure
from reactpy import component, html, use_state, use_effect
import httpx
from pathlib import Path


class Settings:
    """Application configuration settings."""

    HOST: str = "0.0.0.0"
    PORT: int = 3000
    BACKEND_URL: str = "http://localhost:8000"  # Backend API URL
    APP_TITLE: str = "B.R.U.T.U.S."


settings = Settings()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            set_loading(True)
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First call save-image endpoint
                save_response = await client.post(
                    f"{settings.BACKEND_URL}/analyze/save-image"
                )
                if save_response.status_code != 200:
                    set_error("Failed to process image")
                    return
                # Update timestamp to force image refresh
                set_timestamp(int(time.time()))
        except Exception as e:
            set_error(str(e))
        finally:
            set_loading(False)

    use_effect(process_image)

    if loading:
        return html.div(
            {"class": "loading"},
            [html.div({"class": "spinner"}), html.p("Processing image...")],
        )

    if error:
        return html.div({"class": "error"}, f"Error: {error}")

    # Add timestamp to URL to prevent caching
    image_url = f"{settings.BACKEND_URL}/images/analyzed?t={timestamp}"

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

    async def fetch_analysis_results():
        try:
            set_loading(True)
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First ensure image is processed
                save_response = await client.post(
                    f"{settings.BACKEND_URL}/analyze/save-image"
                )
                if save_response.status_code != 200:
                    set_error("Failed to process image")
                    return

                # Then fetch analysis results
                labels_response = await client.get(
                    f"{settings.BACKEND_URL}/analyze/image"
                )
                faces_response = await client.get(
                    f"{settings.BACKEND_URL}/analyze/faces"
                )
                facial_recognition = await client.get(
                    f"{settings.BACKEND_URL}/analyze/facial-recognition"
                )

                if all(
                    r.status_code == 200
                    for r in [labels_response, faces_response, facial_recognition]
                ):
                    labels_data = (
                        labels_response.json().get("data", {}).get("labels", [])
                    )
                    faces_data = faces_response.json().get("data", {})
                    recognition_data = facial_recognition.json().get("data", {})

                    set_results(
                        {
                            "labels": labels_data,
                            "faces": faces_data,
                            "facial_recognition": facial_recognition.json(),
                        }
                    )
                else:
                    set_error("Failed to fetch analysis results")
        except Exception as e:
            set_error(str(e))
        finally:
            set_loading(False)

    use_effect(fetch_analysis_results)

    if loading:
        return html.div(
            {"class": "loading"},
            [html.div({"class": "spinner"}), html.p("Loading analysis results...")],
        )

    if error:
        return html.div({"class": "error"}, f"Error: {error}")

    def format_label(label):
        """Format label information"""
        name = label.get("Name", "Unknown")
        confidence = label.get("Confidence", 0)
        return f"{name} ({confidence:.2f}%)"

    def format_face_details(face):
        """Format face details"""
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
            html.link({"rel": "stylesheet", "href": "/static/styles/main.css"}),
            Header(),
            AnalyzedImage(),
            AnalysisResults(),
        ],
    )


# Configure the app with the root component
configure(app, App)

# Add CSS styles
@app.get("/static/styles.css")
async def get_styles():
    return """
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        font-family: Arial, sans-serif;
    }

    .header {
        text-align: center;
        margin-bottom: 30px;
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 8px;
    }

    .header h1 {
        margin: 0;
        color: #333;
    }

    .image-container {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .analyzed-image {
        max-width: 800px;
        width: 100%;
        height: auto;
        border-radius: 4px;
        margin: 20px 0;
    }

    .result-section {
        margin: 20px 0;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .result-section h3 {
        color: #333;
        margin-top: 0;
    }

    .loading {
        text-align: center;
        padding: 20px;
        color: #666;
    }

    .spinner {
        width: 40px;
        height: 40px;
        margin: 20px auto;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .faces-list {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .face-detail {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

.   face-detail h4 {
        margin-top: 0;
        margin-bottom: 10px;
        color: #495057;
    }

    .face-detail p {
        margin: 5px 0;
        color: #212529;
    }

    .error {
        color: #dc3545;
        padding: 20px;
        text-align: center;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    ul {
        list-style-type: none;
        padding: 0;
    }

    li {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }

    li:last-child {
        border-bottom: none;
    }
    """


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
