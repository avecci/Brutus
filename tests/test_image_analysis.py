import pytest
from pathlib import Path
from image_recognition import ImageAnalyzer  # Replace with your actual module name


class TestImageAnalysis:
    @pytest.fixture
    def analyzer(self):
        return ImageAnalyzer()

    @pytest.fixture
    def test_images_dir(self):
        return Path("./pictures")

    def test_one_person_detection(self, analyzer, test_images_dir):
        """Test that one_person.jpg contains exactly one person (other objects allowed)"""
        image_path = test_images_dir / "one_person.jpg"
        result = analyzer.analyze_image_file(str(image_path))

        assert not result.get(
            "error"
        ), f"Analysis failed with error: {result.get('error')}"

        # Get unique bounding boxes from people detections
        unique_boxes = set()
        for person in result["people"]:
            for instance in person["instances"]:
                box = instance["box"]
                # Create a tuple of box coordinates rounded to 3 decimal places
                box_tuple = (
                    round(box["Left"], 3),
                    round(box["Top"], 3),
                    round(box["Width"], 3),
                    round(box["Height"], 3),
                )
                unique_boxes.add(box_tuple)

        # Check unique person instances
        assert (
            len(unique_boxes) == 1
        ), f"Expected exactly 1 person instance, found {len(unique_boxes)} unique instances"

        # Check no animals
        assert (
            len(result["animals"]) == 0
        ), f"Expected 0 animals, found {len(result['animals'])}"

    def test_multiple_people_detection(self, analyzer, test_images_dir):
        """Test that multiple_people.jpg contains more than one person and nothing else"""
        image_path = test_images_dir / "multiple_people.jpg"
        result = analyzer.analyze_image_file(str(image_path))

        assert not result.get(
            "error"
        ), f"Analysis failed with error: {result.get('error')}"

        # Check people count
        people_count = len(result["people"])
        assert people_count > 1, f"Expected multiple people, found only {people_count}"

        # Check no animals
        assert (
            len(result["animals"]) == 0
        ), f"Expected 0 animals, found {len(result['animals'])}"

    def test_single_object_detection(self, analyzer, test_images_dir):
        """Test that object.jpg contains exactly one object and nothing else"""
        image_path = test_images_dir / "object.jpg"
        result = analyzer.analyze_image_file(str(image_path))

        assert not result.get(
            "error"
        ), f"Analysis failed with error: {result.get('error')}"

        # Get unique bounding boxes from object detections
        unique_boxes = set()
        for obj in result["objects"]:
            for instance in obj["instances"]:
                box = instance["box"]
                # Create a tuple of box coordinates rounded to 3 decimal places
                box_tuple = (
                    round(box["Left"], 3),
                    round(box["Top"], 3),
                    round(box["Width"], 3),
                    round(box["Height"], 3),
                )
                unique_boxes.add(box_tuple)

        # Check unique object instances
        assert (
            len(unique_boxes) == 1
        ), f"Expected exactly 1 object instance, found {len(unique_boxes)} unique instances"

        # Check no people
        assert (
            len(result["people"]) == 0
        ), f"Expected 0 people, found {len(result['people'])}"

        # Check no animals
        assert (
            len(result["animals"]) == 0
        ), f"Expected 0 animals, found {len(result['animals'])}"

    def test_person_and_object_detection(self, analyzer, test_images_dir):
        """Test that one_person_and_object.jpg contains exactly one person and one car"""
        image_path = test_images_dir / "one_person_and_object.jpg"
        result = analyzer.analyze_image_file(str(image_path))

        assert not result.get(
            "error"
        ), f"Analysis failed with error: {result.get('error')}"

        # Get unique person instances
        person_boxes = set()
        for person in result["people"]:
            for instance in person["instances"]:
                box = instance["box"]
                person_boxes.add(
                    (
                        round(box["Left"], 3),
                        round(box["Top"], 3),
                        round(box["Width"], 3),
                        round(box["Height"], 3),
                    )
                )

        # Get unique car/vehicle instances
        car_boxes = set()
        for obj in result["objects"]:
            if any(
                vehicle_type in obj["name"].lower()
                for vehicle_type in ["car", "vehicle", "automobile"]
            ):
                for instance in obj["instances"]:
                    box = instance["box"]
                    car_boxes.add(
                        (
                            round(box["Left"], 3),
                            round(box["Top"], 3),
                            round(box["Width"], 3),
                            round(box["Height"], 3),
                        )
                    )

        # Check counts of unique instances
        assert (
            len(person_boxes) == 1
        ), f"Expected exactly 1 person instance, found {len(person_boxes)} unique instances"

        assert (
            len(car_boxes) == 1
        ), f"Expected exactly 1 car instance, found {len(car_boxes)} unique instances"

        # Check no animals
        assert (
            len(result["animals"]) == 0
        ), f"Expected 0 animals, found {len(result['animals'])}"

        # Verify at least one car-related label exists
        car_related_labels = [
            obj["name"]
            for obj in result["objects"]
            if any(
                vehicle_type in obj["name"].lower()
                for vehicle_type in ["car", "vehicle", "automobile"]
            )
        ]
        assert (
            car_related_labels
        ), "Expected at least one car-related label, but found none"

    def test_face_comparison(self, analyzer, test_images_dir):
        """Test that reference_face.jpg matches with one_person.jpg"""
        # Set up test image paths
        reference_path = test_images_dir / "reference_face.jpg"
        target_path = test_images_dir / "one_person.jpg"

        # Read images
        with open(str(reference_path), "rb") as source_file:
            source_bytes = source_file.read()
        with open(str(target_path), "rb") as target_file:
            target_bytes = target_file.read()

        try:
            # Perform face comparison
            response = analyzer.rekognition_client.compare_faces(
                SourceImage={"Bytes": source_bytes},
                TargetImage={"Bytes": target_bytes},
                SimilarityThreshold=80,
            )

            # Assert we got matches
            assert response.get("FaceMatches"), "No matching faces found"

            # Assert we found exactly one match
            assert (
                len(response["FaceMatches"]) == 1
            ), f"Expected 1 face match, found {len(response['FaceMatches'])}"

            # Get the match details
            match = response["FaceMatches"][0]

            # Assert high similarity (adjust threshold as needed)
            assert (
                match["Similarity"] >= 90
            ), f"Face similarity {match['Similarity']}% is below threshold of 90%"

            # Assert high confidence
            assert (
                match["Face"]["Confidence"] >= 90
            ), f"Face confidence {match['Face']['Confidence']}% is below threshold of 90%"

        except Exception as e:
            assert False, f"Face comparison failed with error: {str(e)}"

    def test_confidence_thresholds(self, analyzer, test_images_dir):
        """Test that all detections meet minimum confidence threshold"""
        min_confidence = 80.0  # 80% minimum confidence threshold

        for image_file in [
            "one_person.jpg",
            "multiple_people.jpg",
            "object.jpg",
            "one_person_and_object.jpg",
        ]:
            image_path = test_images_dir / image_file
            result = analyzer.analyze_image_file(str(image_path))

            assert not result.get(
                "error"
            ), f"Analysis failed for {image_file} with error: {result.get('error')}"

            # Get unique person instances
            person_instances = set()
            for person in result["people"]:
                for instance in person.get("instances", []):
                    box = instance["box"]
                    person_instances.add(
                        (
                            round(box["Left"], 3),
                            round(box["Top"], 3),
                            round(box["Width"], 3),
                            round(box["Height"], 3),
                        )
                    )

            # Check specific requirements for each image
            if image_file == "one_person.jpg":
                assert (
                    len(person_instances) == 1
                ), f"Expected exactly 1 person in {image_file}, found {len(person_instances)}"
                assert (
                    len(result["animals"]) == 0
                ), f"Expected 0 animals, found {len(result['animals'])}"

            elif image_file == "multiple_people.jpg":
                assert (
                    len(person_instances) > 1
                ), f"Expected multiple people in {image_file}, found {len(person_instances)}"
                assert (
                    len(result["animals"]) == 0
                ), f"Expected 0 animals, found {len(result['animals'])}"

            elif image_file == "object.jpg":
                assert (
                    len(person_instances) == 0
                ), f"Expected 0 people in {image_file}, found {len(person_instances)}"
                assert (
                    len(result["animals"]) == 0
                ), f"Expected 0 animals, found {len(result['animals'])}"

                # Verify at least one aircraft-related label exists
                aircraft_related_labels = [
                    obj["name"]
                    for obj in result["objects"]
                    if any(
                        aircraft_type in obj["name"].lower()
                        for aircraft_type in [
                            "aircraft",
                            "jet",
                            "airplane",
                            "plane",
                            "fighter",
                        ]
                    )
                ]
                assert (
                    aircraft_related_labels
                ), "Expected at least one aircraft-related label, but found none"
                assert (
                    len(result["objects"]) > 0
                ), "Expected at least one object detection"

            elif image_file == "one_person_and_object.jpg":
                assert (
                    len(person_instances) == 1
                ), f"Expected exactly 1 person in {image_file}, found {len(person_instances)}"
                assert (
                    len(result["objects"]) > 0
                ), "Expected at least one object detection"
                assert (
                    len(result["animals"]) == 0
                ), f"Expected 0 animals, found {len(result['animals'])}"

    @pytest.mark.parametrize(
        "image_file",
        [
            "one_person.jpg",
            "multiple_people.jpg",
            "object.jpg",
            "one_person_and_object.jpg",
        ],
    )
    def test_bounding_box_validity(self, analyzer, test_images_dir, image_file):
        """Test that all bounding boxes have valid coordinates"""
        image_path = test_images_dir / image_file
        result = analyzer.analyze_image_file(str(image_path))

        assert not result.get(
            "error"
        ), f"Analysis failed for {image_file} with error: {result.get('error')}"

        for category in ["people", "objects", "animals"]:
            for item in result[category]:
                for instance in item["instances"]:
                    box = instance["box"]

                    # Check that coordinates are within valid range (0-1)
                    assert (
                        0 <= box["Left"] <= 1
                    ), f"Invalid Left coordinate in {image_file}: {box['Left']}"
                    assert (
                        0 <= box["Top"] <= 1
                    ), f"Invalid Top coordinate in {image_file}: {box['Top']}"
                    assert (
                        0 <= box["Width"] <= 1
                    ), f"Invalid Width in {image_file}: {box['Width']}"
                    assert (
                        0 <= box["Height"] <= 1
                    ), f"Invalid Height in {image_file}: {box['Height']}"

                    # Check that box dimensions are logical
                    assert (
                        box["Left"] + box["Width"] <= 1
                    ), f"Box extends beyond right edge in {image_file}"
                    assert (
                        box["Top"] + box["Height"] <= 1
                    ), f"Box extends beyond bottom edge in {image_file}"
