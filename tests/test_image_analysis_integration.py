import pytest
from pathlib import Path
from image_recognition import ImageAnalyzer


@pytest.mark.integration
class TestImageAnalysisIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.analyzer = ImageAnalyzer()
        self.test_images_dir = Path("tests/test_data/images")

    @pytest.fixture
    def get_image_path(self):
        """Helper fixture to get image paths and ensure they exist"""

        def _get_path(image_name):
            path = self.test_images_dir / image_name
            assert path.exists(), f"Test image not found: {image_name}"
            return path

        return _get_path

    def get_unique_boxes(self, items):
        """Helper method to extract unique bounding boxes"""
        unique_boxes = set()
        for item in items:
            box = item.get("bounding_box", {})
            if box:  # Only add if box exists
                box_tuple = (
                    round(box["Left"], 3),
                    round(box["Top"], 3),
                    round(box["Width"], 3),
                    round(box["Height"], 3),
                )
                unique_boxes.add(box_tuple)
        return unique_boxes

    def assert_no_error(self, result):
        """Helper method to check for analysis errors"""
        assert not result.get(
            "error"
        ), f"Analysis failed with error: {result.get('error')}"

    @pytest.mark.parametrize(
        "image_name,min_people,min_animals,min_objects",
        [
            ("one_person.jpg", 1, 0, 0),  # At least 1 person
            ("multiple_people.jpg", 3, 0, 0),  # Should find exactly 3 people
            ("object.jpg", 0, 0, 1),  # At least 1 object, no people/animals
            ("one_person_and_object.jpg", 1, 0, 1),  # At least 1 person and 1 object
            ("animal.jpg", 0, 1, 0),  # At least 1 animal, no people/objects
        ],
    )
    def test_detection_counts(
        self, get_image_path, image_name, min_people, min_animals, min_objects
    ):
        """Test detection counts for various image types"""
        image_path = get_image_path(image_name)
        result = self.analyzer.analyze_image_file(str(image_path))

        self.assert_no_error(result)

        # Debug logging
        print(f"\nAnalysis result for {image_name}:")
        print(f"Raw people detections: {result['people']}")

        people_boxes = self.get_unique_boxes(result["people"])
        animal_boxes = self.get_unique_boxes(result["animals"])
        object_boxes = self.get_unique_boxes(result["objects"])

        print(f"Unique people boxes found: {people_boxes}")

        # Check minimum counts while allowing for additional detections
        assert len(people_boxes) >= min_people, (
            f"Expected at least {min_people} people, found {len(people_boxes)}.\n"
            f"Raw detections: {result['people']}"
        )
        assert (
            len(animal_boxes) >= min_animals
        ), f"Expected at least {min_animals} animals, found {len(animal_boxes)}"
        assert (
            len(object_boxes) >= min_objects
        ), f"Expected at least {min_objects} objects, found {len(object_boxes)}"

        # Check that we don't detect people/animals where there shouldn't be any
        if min_people == 0:
            assert (
                len(people_boxes) == 0
            ), f"Found {len(people_boxes)} people in {image_name} where there should be none"
        if min_animals == 0:
            assert (
                len(animal_boxes) == 0
            ), f"Found {len(animal_boxes)} animals in {image_name} where there should be none"

    @pytest.mark.parametrize(
        "image_name,min_people,min_animals,min_objects",
        [
            ("one_person.jpg", 1, 0, 0),  # At least 1 person
            ("multiple_people.jpg", 3, 0, 0),  # Should find exactly 3 people
            ("object.jpg", 0, 0, 1),  # At least 1 object, no people/animals
            ("one_person_and_object.jpg", 1, 0, 1),  # At least 1 person and 1 object
            ("animal.jpg", 0, 1, 0),  # At least 1 animal, no people/objects
        ],
    )
    def test_bounding_box_validity(
        self, get_image_path, image_name, min_people, min_animals, min_objects
    ):
        """Test bounding box coordinate validity"""
        image_path = get_image_path(image_name)
        result = self.analyzer.analyze_image_file(str(image_path))

        self.assert_no_error(result)

        for category in ["people", "objects", "animals"]:
            for item in result[category]:
                box = item.get("bounding_box", {})
                if box:
                    # Check coordinate ranges
                    assert (
                        0 <= box["Left"] <= 1
                    ), f"Invalid Left coordinate in {image_name}: {box['Left']}"
                    assert (
                        0 <= box["Top"] <= 1
                    ), f"Invalid Top coordinate in {image_name}: {box['Top']}"
                    assert (
                        0 <= box["Width"] <= 1
                    ), f"Invalid Width in {image_name}: {box['Width']}"
                    assert (
                        0 <= box["Height"] <= 1
                    ), f"Invalid Height in {image_name}: {box['Height']}"

                    # Check box dimensions
                    assert (
                        box["Left"] + box["Width"] <= 1
                    ), f"Box extends beyond right edge in {image_name}"
                    assert (
                        box["Top"] + box["Height"] <= 1
                    ), f"Box extends beyond bottom edge in {image_name}"
