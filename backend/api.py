"""FastAPI wrapper for Brutus backend."""
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Union

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from image_recognition import BrutusEyes
from logging_utils import setup_logger

logger = setup_logger(__name__)


class Settings:
    """Application configuration settings."""

    HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000


settings = Settings()

app = FastAPI(title="Brutus Eyes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

brutus_eyes = BrutusEyes()

# Define default paths
DEFAULT_INPUT_PATH = Path("input/input_image.jpg")
DEFAULT_OUTPUT_PATH = Path("output/analyzed_image.jpg")
DEFAULT_KNOWN_FACES_DIR = Path("reference_library/")
OUTPUT_DIR = Path(__file__).parent / "output"
ANALYZED_IMAGE_PATH = OUTPUT_DIR / "analyzed_image.jpg"


@app.get("/", status_code=200)
async def root() -> Dict[str, Any]:
    """Root endpoint providing API information."""
    logger.info("Accessing root endpoint")
    return JSONResponse(
        content={
            "name": "Brutus backend API",
            "version": "1.0.0",
            "status": "online",
            "description": "Endpoints for Brutus functionalities provided by backend functions.",
            "endpoints": {
                "health": "/health",
                "analyze_image": "/analyze/image",
                "analyze_faces": "/analyze/faces",
                "facial_recognition": "/analyze/facial-recognition",
                "analyze_all": "/analyze/all",
                "save_analyzed_image": "/analyze/save-image",
                "upload_image": "/upload/image",
            },
        }
    )


@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "Healthy"}


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and save a JPG image."""
    try:
        logger.info(
            f"Attempting to upload image: {urllib.parse.quote(str(file.filename))}"
        )
        if not file:
            logger.error("No file uploaded")
            raise HTTPException(status_code=400, detail="No file uploaded")

        if file.content_type not in ["image/jpeg", "image/jpg"]:
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400, detail="Only JPG/JPEG files are allowed"
            )

        input_dir = Path("input")
        input_dir.mkdir(exist_ok=True)
        file_path = input_dir / "input_image.jpg"

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Successfully saved uploaded image to {file_path}")
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "data": {
                    "filename": str(file_path),
                },
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")
    finally:
        file.file.close()


@app.get("/analyze/image", status_code=200)
async def analyze_image(
    input_path: Union[str, Path] = str(DEFAULT_INPUT_PATH)
) -> Dict[str, Any]:
    """Analyze image and return detected labels."""
    try:
        logger.info(f"Analyzing image at path: {urllib.parse.quote(str(input_path))}")
        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input image not found: {input_path}")
            raise HTTPException(
                status_code=404, detail=f"404: Input image not found at {input_path}"
            )

        labels = brutus_eyes.detect_labels_in_image(str(input_path))
        logger.info(f"Successfully analyzed image, found {len(labels)} labels")

        return JSONResponse(
            content={
                "status": "success",
                "data": {"input_image": str(input_path), "labels": labels},
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting labels: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error detecting labels: {str(e)}")


@app.get("/analyze/faces", status_code=200)
async def detect_faces(
    input_path: Union[str, Path] = str(DEFAULT_INPUT_PATH)
) -> Dict[str, Any]:
    """Detect faces and their attributes in an image."""
    try:
        logger.info(
            f"Starting face detection for image: {urllib.parse.quote(input_path)}"
        )
        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input image not found: {input_path}")
            raise HTTPException(
                status_code=404, detail=f"404: Input image not found at {input_path}"
            )

        face_details = brutus_eyes.detect_and_return_face_details(str(input_path))
        logger.info(f"Successfully detected {len(face_details)} faces in image")

        return JSONResponse(
            content={
                "status": "success",
                "data": {"input_image": str(input_path), "faces": face_details},
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")


@app.get("/analyze/facial-recognition")
async def facial_recognition(
    input_path: Union[str, Path] = str(DEFAULT_INPUT_PATH),
    reference_dir: Union[str, Path] = str(DEFAULT_KNOWN_FACES_DIR),
    similarity_threshold: float = 85,
) -> Dict[str, Any]:
    """Compare faces in input image with reference library."""
    try:
        logger.info(
            f"Starting facial recognition - Input: {input_path}, Reference dir: {reference_dir}"
        )
        input_path = Path(input_path)
        reference_dir = Path(reference_dir)

        if not input_path.exists():
            logger.error(f"Input image not found: {input_path}")
            raise HTTPException(
                status_code=404, detail=f"404: Input image not found at {input_path}"
            )

        if not reference_dir.exists():
            logger.error(f"Reference directory not found: {reference_dir}")
            raise HTTPException(
                status_code=404,
                detail=f"Reference library directory not found at {reference_dir}",
            )

        matches = brutus_eyes.compare_faces_with_library(
            str(input_path), str(reference_dir), similarity_threshold
        )

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "input_image": str(input_path),
                    "reference_dir": str(reference_dir),
                    "similarity_threshold": similarity_threshold,
                    "matches": matches,
                },
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")


@app.get("/analyze/all", status_code=200)
async def analyze_all(
    input_path: Union[str, Path] = str(DEFAULT_INPUT_PATH),
    known_faces_dir: Union[str, Path] = str(DEFAULT_KNOWN_FACES_DIR),
) -> Dict[str, Any]:
    """Perform comprehensive analysis of an image."""
    try:
        logger.info(
            f"Starting comprehensive analysis - Input: {urllib.parse.quote(str(input_path))}, Known faces dir: {urllib.parse.quote(str(known_faces_dir))}"
        )
        input_path = Path(input_path)
        known_faces_dir = Path(known_faces_dir)

        if not input_path.exists():
            logger.error(f"Input image not found: {input_path}")
            raise HTTPException(
                status_code=404, detail=f"404: Input image not found at {input_path}"
            )

        if not known_faces_dir.exists():
            logger.error(f"Known faces directory not found: {known_faces_dir}")
            raise HTTPException(
                status_code=404,
                detail=f"404: Known faces directory not found at {known_faces_dir}",
            )

        logger.info("Detecting labels...")
        labels = brutus_eyes.detect_labels_in_image(str(input_path))
        logger.info(f"Found {len(labels)} labels")

        logger.info("Detecting faces...")
        faces = brutus_eyes.detect_and_return_face_details(str(input_path))
        logger.info(f"Found {len(faces)} faces")

        logger.info("Performing facial recognition...")
        face_matches = brutus_eyes.compare_faces_with_library(
            str(input_path), str(known_faces_dir)
        )
        logger.info(f"Found {len(face_matches)} face matches")

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "input_image": str(input_path),
                    "labels": labels,
                    "faces": faces,
                    "face_matches": face_matches,
                },
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error performing analysis: {str(e)}"
        )


@app.post("/analyze/save-image", status_code=201)
async def save_analyzed_image(
    input_path: Union[str, Path] = str(DEFAULT_INPUT_PATH),
    output_path: Union[str, Path] = str(DEFAULT_OUTPUT_PATH),
    known_faces_dir: Union[str, Path] = str(DEFAULT_KNOWN_FACES_DIR),
) -> Dict[str, Any]:
    """Analyze image and save the result with bounding boxes."""
    try:
        logger.info(
            f"Starting image analysis and save - Input: {urllib.parse.quote(str(input_path))}, Output: {urllib.parse.quote(str(output_path))}"
        )
        input_path = Path(input_path)
        known_faces_dir = Path(known_faces_dir)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error(f"Input image not found: {input_path}")
            raise HTTPException(
                status_code=404, detail=f"404: Input image not found at {input_path}"
            )

        if not known_faces_dir.exists():
            logger.error(f"Known faces directory not found: {known_faces_dir}")
            raise HTTPException(
                status_code=404,
                detail=f"404: Known faces directory not found at {known_faces_dir}",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path.parent}")

        logger.info("Drawing bounding boxes...")
        marked_image = brutus_eyes.draw_bounding_boxes(
            str(input_path), str(known_faces_dir)
        )
        marked_image.save(str(output_path))

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "input_image": str(input_path),
                    "output_image": str(output_path),
                },
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving analyzed image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error saving analyzed image: {str(e)}"
        )


@app.get("/images/analyzed")
async def get_analyzed_image():
    """Return the analyzed image from the output directory.

    Returns a 404 if the image doesn't exist.


    Test with
    > http://localhost:8000/images/analyzed
    """
    try:
        logger.info("Attempting to retrieve analyzed image")
        if not ANALYZED_IMAGE_PATH.exists():
            logger.error(f"Analyzed image not found at {ANALYZED_IMAGE_PATH}")
            raise HTTPException(
                status_code=404,
                detail="404: Analyzed image not found. Please ensure image analysis has been completed.",
            )

        logger.info("Successfully retrieved analyzed image")
        return FileResponse(
            ANALYZED_IMAGE_PATH, media_type="image/jpeg", filename="analyzed_image.jpg"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analyzed image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error retrieving analyzed image: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.BACKEND_PORT)
