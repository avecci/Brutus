from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict, Any

from image_recognition import BrutusEyes

app = FastAPI(title="Brutus Eyes API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize BrutusEyes
brutus_eyes = BrutusEyes()

# Define default paths
DEFAULT_INPUT_PATH = Path("input/input_image.jpg")
DEFAULT_OUTPUT_PATH = Path("output/analyzed_image.jpg")
DEFAULT_KNOWN_FACES_DIR = Path("reference_library/")
OUTPUT_DIR = Path(__file__).parent / "output"
ANALYZED_IMAGE_PATH = OUTPUT_DIR / "analyzed_image.jpg"


@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint."""
    return {"status": "Healthy"}


@app.get("/analyze/image", status_code=200)
async def analyze_image(input_path: str = str(DEFAULT_INPUT_PATH)) -> Dict[str, Any]:
    """
    Analyze an image and return detected labels.

    curl http://localhost:8000/analyze/image

    Args:
        input_path: Path to the input image (optional)

    Returns:
        Dictionary containing detected labels and their details
    """
    try:
        # Ensure input file exists
        input_path = Path(input_path)
        if not input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Input image not found at {input_path}"
            )

        # Get labels from Rekognition
        labels = brutus_eyes.detect_labels_in_image(str(input_path))

        return JSONResponse(
            content={
                "status": "success",
                "data": {"input_image": str(input_path), "labels": labels},
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/faces", status_code=200)
async def detect_faces(input_path: str = str(DEFAULT_INPUT_PATH)) -> Dict[str, Any]:
    """
    Detect faces and their attributes in an image.

    Try with

    > curl http://localhost:8000/analyze/faces

    Args:
        input_path: Path to the input image (optional)

    Returns:
        Dictionary containing detected faces and their attributes
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Input image not found at {input_path}"
            )

        face_details = brutus_eyes.detect_and_return_face_details(str(input_path))

        return JSONResponse(
            content={
                "status": "success",
                "data": {"input_image": str(input_path), "faces": face_details},
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/facial-recognition")
async def facial_recognition(
    input_path: str = str(DEFAULT_INPUT_PATH),
    reference_dir: str = "reference_library",
    similarity_threshold: float = 85,
) -> Dict[str, Any]:
    """
    Compare faces in input image with reference library.

    Try with:
    > curl http://localhost:8000/analyze/facial-recognition


    or with custom parameters:

    > curl "http://localhost:8000/analyze/facial-recognition?similarity_threshold=85"


    Args:
        input_path: Path to the input image (optional, defaults to input/input_image.jpg)
        reference_dir: Path to reference library directory (optional, defaults to reference_library)
        similarity_threshold: Minimum similarity percentage (default: 80)

    Returns:
        Dictionary containing face matches
    """
    try:
        input_path = Path(input_path)
        reference_dir = Path(reference_dir)

        if not input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Input image not found at {input_path}"
            )

        if not reference_dir.exists():
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/all", status_code=200)
async def analyze_all(
    input_path: str = str(DEFAULT_INPUT_PATH),
    known_faces_dir: str = str(DEFAULT_KNOWN_FACES_DIR),
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of an image (labels, faces, text).

    Args:
        input_path: Path to the input image (optional)
        known_faces_dir: Path to known faces directory (optional)

    Returns:
        Dictionary containing all analysis results
    """
    try:
        input_path = Path(input_path)
        known_faces_dir = Path(known_faces_dir)

        if not input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Input image not found at {input_path}"
            )

        if not known_faces_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Known faces directory not found at {known_faces_dir}",
            )

        # Get all types of analysis
        labels = brutus_eyes.detect_labels_in_image(str(input_path))
        faces = brutus_eyes.detect_and_return_face_details(str(input_path))
        face_matches = brutus_eyes.compare_faces_with_library(
            str(input_path), str(known_faces_dir)
        )

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/save-image", status_code=201)
async def save_analyzed_image(
    input_path: str = str(DEFAULT_INPUT_PATH),
    output_path: str = str(DEFAULT_OUTPUT_PATH),
    known_faces_dir: str = str(DEFAULT_KNOWN_FACES_DIR),
) -> Dict[str, Any]:
    """
    Analyze image and save the result with bounding boxes.

    Basic usage:
    curl -X POST http://localhost:8000/analyze/save-image

    Test with a specific input image
    curl "http://localhost:8000/analyze/image?input_path=input/different_image.jpg"

    Save analysis with custom paths
    curl -X POST "http://localhost:8000/analyze/save-image?input_path=input/different_image.jpg&output_path=output/custom_output.jpg"


    Args:
        input_path: Path to the input image (optional)
        output_path: Path to save the analyzed image (optional)
        known_faces_dir: Path to the directory with known faces (optional)

    Returns:
        Dictionary containing paths to input and output images
    """
    try:
        # Ensure input file and known_faces_dir exist
        input_path = Path(input_path)
        known_faces_dir = Path(known_faces_dir)
        output_path = Path(output_path)

        if not input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Input image not found at {input_path}"
            )

        if not known_faces_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Known faces directory not found at {known_faces_dir}",
            )

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Draw bounding boxes and save image
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/analyzed")
async def get_analyzed_image():
    """
    Return the analyzed image from the output directory.
    Returns a 404 if the image doesn't exist.


    Test with
    > http://localhost:8000/images/analyzed
    """
    if not ANALYZED_IMAGE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Analyzed image not found. Please ensure image analysis has been completed.",
        )

    return FileResponse(
        ANALYZED_IMAGE_PATH, media_type="image/jpeg", filename="analyzed_image.jpg"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
