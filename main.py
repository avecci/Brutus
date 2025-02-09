from image_recognition import ImageAnalyzer

def main():
    try:
        analyzer = ImageAnalyzer()

        # Set up paths
        input_image = "input/input_image.jpg"
        known_faces_dir = "reference_library/"
        output_path = "output/analyzed_image.jpg"

        print(f"\nAnalyzing image: {input_image}")

        # First try to identify any known people
        print("\nChecking for known people...")
        face_results = analyzer.compare_faces_with_library(input_image, known_faces_dir)

        # Read image bytes once for all operations
        with open(input_image, "rb") as image_file:
            image_bytes = image_file.read()

        # Get general content analysis
        content_results = analyzer.detect_labels_in_image(image_bytes)
        
        # Get detailed face analysis
        face_details = analyzer.detect_face_details(image_bytes)

        # Print comprehensive analysis results
        print("\n=== Image Analysis Results ===")

        # 1. Print people detection results
        if content_results.get("people"):
            print("\n👥 People Detection:")
            for person in content_results["people"]:
                print(f"- Person detected (Confidence: {person['confidence']})")

            # 2. Print detailed face attributes for each detected face
            if face_details:
                print("\n👤 Face Details:")
                for i, face in enumerate(face_details, 1):
                    print(f"\nFace {i}:")
                    
                    # Age Range
                    if 'AgeRange' in face:
                        print(f"  Age Range: {face['AgeRange']['Low']}-{face['AgeRange']['High']} years")
                    
                    # Gender
                    if 'Gender' in face:
                        print(f"  Gender: {face['Gender']['Value']} "
                              f"(Confidence: {face['Gender']['Confidence']:.2f}%)")
                    
                    # Emotions - Show top 3 emotions
                    if 'Emotions' in face:
                        print("  Emotions:")
                        sorted_emotions = sorted(face['Emotions'], 
                                              key=lambda x: x['Confidence'], 
                                              reverse=True)[:3]
                        for emotion in sorted_emotions:
                            print(f"    - {emotion['Type'].lower().capitalize()}: "
                                  f"{emotion['Confidence']:.2f}%")
                    
                    # Facial Features
                    features = {
                        'Smile': 'Smiling',
                        'EyesOpen': 'Eyes open',
                        'MouthOpen': 'Mouth open',
                        'Eyeglasses': 'Wearing glasses',
                        'Sunglasses': 'Wearing sunglasses'
                    }
                    
                    print("  Facial Features:")
                    for key, description in features.items():
                        if key in face:
                            value = face[key]['Value']
                            confidence = face[key]['Confidence']
                            print(f"    - {description}: {'Yes' if value else 'No'} "
                                  f"({confidence:.2f}%)")
                    
                    # Quality metrics
                    if 'Quality' in face:
                        print("  Image Quality:")
                        print(f"    - Brightness: {face['Quality']['Brightness']:.2f}")
                        print(f"    - Sharpness: {face['Quality']['Sharpness']:.2f}")

        # 3. Print animal detection results
        if content_results.get("animals"):
            print("\n🐾 Animals Detected:")
            for animal in content_results["animals"]:
                print(f"- {animal['name']} (Confidence: {animal['confidence']})")

        # 4. Print clothing and accessories
        if content_results.get("objects"):
            clothing_items = [obj for obj in content_results["objects"] 
                            if any(item in obj['name'].lower() 
                                  for item in ['clothing', 'shirt', 'pants', 'dress', 
                                             'jacket', 'hat', 'accessories', 'shoes'])]
            
            if clothing_items:
                print("\n👔 Clothing and Accessories:")
                for item in clothing_items:
                    print(f"- {item['name']} (Confidence: {item['confidence']})")

        # Draw boxes on the image
        print("\nGenerating annotated image...")
        if face_results.get("matches") and any(match["similarity"] > 0 
                                             for match in face_results["matches"]):
            # If known faces were found, draw face recognition boxes
            analyzer.draw_identified_faces(input_image, face_results["matches"], 
                                        output_path)
        else:
            # If no known faces, draw general detection boxes
            analyzer.draw_bounding_boxes(input_image, content_results, output_path)

        print(f"Annotated image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
