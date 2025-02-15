import pytest
from pathlib import Path
from image_recognition import BrutusSees
from PIL import Image


@pytest.mark.integration
class TestImageAnalysisIntegration:
    """
    Test the overall functionality of image analysis.
    Compare output of methods to known images.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.analyzer = BrutusSees()
        self.test_images_dir = Path("tests/test_data/images")
        self.known_faces_dir = self.test_images_dir / "reference"

    @pytest.fixture
    def get_image_path(self):
        """Helper fixture to get image paths and ensure they exist"""

        def _get_path(image_name):
            path = self.test_images_dir / image_name
            assert path.exists(), f"Test image not found: {image_name}"
            return str(path)

        return _get_path

    def test_animal_detection(self, get_image_path):
        """Test animal.jpg - should detect exactly 1 dog, no humans"""
        image_path = get_image_path("animal.jpg")

        # Check for human face detection (should be none)
        faces = self.analyzer.detect_and_return_face_details(image_path)
        assert (
            faces is None or len(faces.get("faces", [])) == 0
        ), "No humans should be detected"

        # Check label detection
        labels = self.analyzer.detect_labels_in_image(image_path)
        animal_instances = sum(
            len(label.get("Instances", []))
            for label in labels
            if label["Name"] in ["Animal", "Dog", "Pet"]
        )
        assert animal_instances == 1, "Exactly one animal should be detected"

    def test_multiple_people_detection(self, get_image_path):
        """Test multiple_people.jpg - should detect exactly 3 people, no animals"""
        image_path = get_image_path("multiple_people.jpg")

        # Check face detection
        faces = self.analyzer.detect_and_return_face_details(image_path)
        assert (
            faces is not None and len(faces["faces"]) == 3
        ), "Exactly three faces should be detected"

        # Check no animals
        labels = self.analyzer.detect_labels_in_image(image_path)
        animal_labels = [
            label
            for label in labels
            if label["Name"] in ["Animal", "Dog", "Pet", "Cat", "Bird"]
        ]
        assert len(animal_labels) == 0, "No animals should be detected"

    def test_object_detection(self, get_image_path):
        """Test object.jpg - should detect exactly 1 airplane, no humans/animals"""
        image_path = get_image_path("object.jpg")

        # Check no faces
        faces = self.analyzer.detect_and_return_face_details(image_path)
        assert (
            faces is None or len(faces.get("faces", [])) == 0
        ), "No humans should be detected"

        # Check labels
        labels = self.analyzer.detect_labels_in_image(image_path)
        airplane_instances = sum(
            len(label.get("Instances", []))
            for label in labels
            if label["Name"] in ["Airplane", "Aircraft", "Plane"]
        )
        assert airplane_instances == 1, "Exactly one airplane should be detected"

        # Verify no animals
        animal_labels = [
            label
            for label in labels
            if label["Name"] in ["Animal", "Dog", "Pet", "Cat", "Bird"]
        ]
        assert len(animal_labels) == 0, "No animals should be detected"

    def test_one_person_detection(self, get_image_path):
        """Test one_person.jpg - should detect exactly 1 person, no animals"""
        image_path = get_image_path("one_person.jpg")

        # Check face detection
        faces = self.analyzer.detect_and_return_face_details(image_path)
        assert (
            faces is not None and len(faces["faces"]) == 1
        ), "Exactly one face should be detected"

        # Check no animals
        labels = self.analyzer.detect_labels_in_image(image_path)
        animal_labels = [
            label
            for label in labels
            if label["Name"] in ["Animal", "Dog", "Pet", "Cat", "Bird"]
        ]
        assert len(animal_labels) == 0, "No animals should be detected"

    def test_person_and_object_detection(self, get_image_path):
        """Test one_person_and_object.jpg - should detect 1 person and at least 1 object (car), no animals"""
        image_path = get_image_path("one_person_and_object.jpg")

        # Check face detection
        faces = self.analyzer.detect_and_return_face_details(image_path)
        assert (
            faces is not None and len(faces["faces"]) == 1
        ), "Exactly one face should be detected"

        # Check labels for car and no animals
        labels = self.analyzer.detect_labels_in_image(image_path)
        car_instances = sum(
            len(label.get("Instances", []))
            for label in labels
            if label["Name"] in ["Car", "Automobile", "Vehicle"]
        )
        assert car_instances >= 1, "At least one car should be detected"

        # Verify no animals
        animal_labels = [
            label
            for label in labels
            if label["Name"] in ["Animal", "Dog", "Pet", "Cat", "Bird"]
        ]
        assert len(animal_labels) == 0, "No animals should be detected"

    def test_person_and_dog_detection(self, get_image_path):
        """Test person_and_dog.jpg - should detect exactly 1 person and 1 dog"""
        image_path = get_image_path("person_and_dog.jpg")

        # Check face detection
        faces = self.analyzer.detect_and_return_face_details(image_path)
        assert (
            faces is not None and len(faces["faces"]) == 1
        ), "Exactly one face should be detected"

        # Check animal detection
        labels = self.analyzer.detect_labels_in_image(image_path)
        animal_instances = sum(
            len(label.get("Instances", []))
            for label in labels
            if label["Name"] in ["Animal", "Dog", "Pet"]
        )
        assert animal_instances == 1, "Exactly one animal should be detected"

    def test_face_recognition(self, get_image_path):
        """Test face recognition between one_person.jpg and reference_person_1.jpg"""
        image_path = get_image_path("one_person.jpg")

        # Test face comparison
        comparison_results = self.analyzer.compare_faces_with_library(
            image_path, str(self.known_faces_dir)
        )

        # Verify person is recognized
        assert comparison_results is not None, "Comparison results should not be None"
        assert "matches" in comparison_results, "Results should contain 'matches' key"
        assert (
            len(comparison_results["matches"]) == 1
        ), "Exactly one face match should be found"
        assert (
            comparison_results["matches"][0]["person"] == "reference_person_1"
        ), "Should match with reference_person_1"
