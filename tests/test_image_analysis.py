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

        assert not result.get('error'), f"Analysis failed with error: {result.get('error')}"
    
        # Get unique bounding boxes from people detections
        unique_boxes = set()
        for person in result['people']:
            for instance in person['instances']:
                box = instance['box']
                # Create a tuple of box coordinates rounded to 3 decimal places
                box_tuple = (
                    round(box['Left'], 3),
                    round(box['Top'], 3),
                    round(box['Width'], 3),
                    round(box['Height'], 3)
                )
                unique_boxes.add(box_tuple)
    
        # Check unique person instances
        assert len(unique_boxes) == 1, (
            f"Expected exactly 1 person instance, found {len(unique_boxes)} unique instances"
        )
    
        # Check no animals
        assert len(result['animals']) == 0, f"Expected 0 animals, found {len(result['animals'])}"


    def test_multiple_people_detection(self, analyzer, test_images_dir):
        """Test that multiple_people.jpg contains more than one person and nothing else"""
        image_path = test_images_dir / "multiple_people.jpg"
        result = analyzer.analyze_image_file(str(image_path))
        
        assert not result.get('error'), f"Analysis failed with error: {result.get('error')}"
        
        # Check people count
        people_count = len(result['people'])
        assert people_count > 1, f"Expected multiple people, found only {people_count}"
        
        # Check no animals
        assert len(result['animals']) == 0, f"Expected 0 animals, found {len(result['animals'])}"
        

    def test_single_object_detection(self, analyzer, test_images_dir):
        """Test that object.jpg contains exactly one object and nothing else"""
        image_path = test_images_dir / "object.jpg"
        result = analyzer.analyze_image_file(str(image_path))

        assert not result.get('error'), f"Analysis failed with error: {result.get('error')}"

        # Get unique bounding boxes from object detections
        unique_boxes = set()
        for obj in result['objects']:
            for instance in obj['instances']:
                box = instance['box']
                # Create a tuple of box coordinates rounded to 3 decimal places
                box_tuple = (
                    round(box['Left'], 3),
                    round(box['Top'], 3),
                    round(box['Width'], 3),
                    round(box['Height'], 3)
                )
                unique_boxes.add(box_tuple)

        # Check unique object instances
        assert len(unique_boxes) == 1, (
            f"Expected exactly 1 object instance, found {len(unique_boxes)} unique instances"
        )

        # Check no people
        assert len(result['people']) == 0, f"Expected 0 people, found {len(result['people'])}"

        # Check no animals
        assert len(result['animals']) == 0, f"Expected 0 animals, found {len(result['animals'])}"


    def test_person_and_object_detection(self, analyzer, test_images_dir):
        """Test that one_person_and_object.jpg contains exactly one person and one car"""
        image_path = test_images_dir / "one_person_and_object.jpg"
        result = analyzer.analyze_image_file(str(image_path))
        
        assert not result.get('error'), f"Analysis failed with error: {result.get('error')}"
        
        # Get unique person instances
        person_boxes = set()
        for person in result['people']:
            for instance in person['instances']:
                box = instance['box']
                person_boxes.add((
                    round(box['Left'], 3),
                    round(box['Top'], 3),
                    round(box['Width'], 3),
                    round(box['Height'], 3)
                ))
        
        # Get unique car/vehicle instances
        car_boxes = set()
        for obj in result['objects']:
            if any(vehicle_type in obj['name'].lower() for vehicle_type in ['car', 'vehicle', 'automobile']):
                for instance in obj['instances']:
                    box = instance['box']
                    car_boxes.add((
                        round(box['Left'], 3),
                        round(box['Top'], 3),
                        round(box['Width'], 3),
                        round(box['Height'], 3)
                    ))
        
        # Check counts of unique instances
        assert len(person_boxes) == 1, (
            f"Expected exactly 1 person instance, found {len(person_boxes)} unique instances"
        )
        
        assert len(car_boxes) == 1, (
            f"Expected exactly 1 car instance, found {len(car_boxes)} unique instances"
        )
        
        # Check no animals
        assert len(result['animals']) == 0, f"Expected 0 animals, found {len(result['animals'])}"
        
        # Verify at least one car-related label exists
        car_related_labels = [
            obj['name'] for obj in result['objects'] 
            if any(vehicle_type in obj['name'].lower() for vehicle_type in ['car', 'vehicle', 'automobile'])
        ]
        assert car_related_labels, "Expected at least one car-related label, but found none"
    
    
    def test_confidence_thresholds(self, analyzer, test_images_dir):
        """Test that all detections meet minimum confidence threshold"""
        MIN_CONFIDENCE = 80.0  # 80% minimum confidence threshold
        
        for image_file in ["one_person.jpg", "multiple_people.jpg", 
                          "object.jpg", "one_person_and_object.jpg"]:
            image_path = test_images_dir / image_file
            result = analyzer.analyze_image_file(str(image_path))
            
            assert not result.get('error'), f"Analysis failed for {image_file} with error: {result.get('error')}"
            
            # Check confidence for all detections
            for category in ['people', 'objects', 'animals']:
                for item in result[category]:
                    confidence = float(item['confidence'].rstrip('%'))
                    assert confidence >= MIN_CONFIDENCE, (
                        f"Low confidence detection ({confidence}%) in {image_file}: "
                        f"{item['name']} in category {category}"
                    )

    @pytest.mark.parametrize("image_file", [
        "one_person.jpg",
        "multiple_people.jpg",
        "object.jpg",
        "one_person_and_object.jpg"
    ])
    def test_bounding_box_validity(self, analyzer, test_images_dir, image_file):
        """Test that all bounding boxes have valid coordinates"""
        image_path = test_images_dir / image_file
        result = analyzer.analyze_image_file(str(image_path))
        
        assert not result.get('error'), f"Analysis failed for {image_file} with error: {result.get('error')}"
        
        for category in ['people', 'objects', 'animals']:
            for item in result[category]:
                for instance in item['instances']:
                    box = instance['box']
                    
                    # Check that coordinates are within valid range (0-1)
                    assert 0 <= box['Left'] <= 1, f"Invalid Left coordinate in {image_file}: {box['Left']}"
                    assert 0 <= box['Top'] <= 1, f"Invalid Top coordinate in {image_file}: {box['Top']}"
                    assert 0 <= box['Width'] <= 1, f"Invalid Width in {image_file}: {box['Width']}"
                    assert 0 <= box['Height'] <= 1, f"Invalid Height in {image_file}: {box['Height']}"
                    
                    # Check that box dimensions are logical
                    assert box['Left'] + box['Width'] <= 1, f"Box extends beyond right edge in {image_file}"
                    assert box['Top'] + box['Height'] <= 1, f"Box extends beyond bottom edge in {image_file}"
