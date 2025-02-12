from image_recognition import ImageAnalyzer
from pprint import pprint

def main():
    # Initialize paths
    input_image = "input/input_image.jpg"
    known_faces_dir = "reference_library/"
    output_path = "output/analyzed_image.jpg"

    # Initialize analyzer
    analyzer = ImageAnalyzer()

    print("\n=== Starting Image Analysis ===\n")

    # 1. Detect labels
    print("Label Detection Results:")
    label_results = analyzer.detect_labels_in_image(input_image)
    label_index = 0
    for label in label_results:
        if 'Instances' in label:
            for instance in label['Instances']:
                print(f"Label {label_index}: {label['Name']} - Confidence: {instance['Confidence']:.1f}%")
                label_index += 1
    print()

    # 2. Compare faces with reference library
    print("Face Comparison Results:")
    face_comparison = analyzer.compare_faces_with_library(input_image, known_faces_dir)
    if face_comparison and "error" not in face_comparison:
        if 'matches' in face_comparison:
            for match in face_comparison['matches']:
                print(f"Face: {match['person']} - Similarity: {match['similarity']:.1f}%")
    print()

    # 3. Get face details
    print("Face Details Results:")
    face_details = analyzer.detect_and_return_face_details(input_image)
    pprint(face_details)
    print()

    # 4. Draw boxes and save image
    # Pass both input_image and known_faces_dir to draw_bounding_boxes
    marked_image = analyzer.draw_bounding_boxes(input_image, known_faces_dir)
    marked_image.save(output_path)
    print(f"Analyzed image saved to: {output_path}")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
