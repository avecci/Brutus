"""Brutus v0: Run analysis so output gets printed on console."""
import os

from image_recognition import BrutusEyes
from speech_generator import BrutusSpeechGenerator


def main():
    """Call image_recognition directly and print results to console."""
    # Initialize paths
    input_image = "input/input_image.jpg"
    known_faces_dir = "reference_library/"
    output_path = "output/analyzed_image.jpg"

    # Initialize classes
    speech_generator = BrutusSpeechGenerator()
    analyzer = BrutusEyes()

    print("\n=== Starting Image Analysis ===\n")

    # 1. Detect labels
    print("\nLabel Detection Results:")
    print("-" * 50)
    label_results = analyzer.detect_labels_in_image(input_image)
    for label in label_results:
        print(f"\nLabel: {label['Name']} (Confidence: {label['Confidence']:.1f}%)")

        if "RelatedLabels" in label:
            print(f"Related labels: {', '.join(label['RelatedLabels'])}")

        # Show parent information
        if label["Parents"]:
            print("Parent labels (from general to specific):")
            for parent in label["Parents"]:
                print(f"  └── {parent['Name']}")

        # Show instances if any
        # if label["Instances"]:
        #    print("Instances found:")
        #    for i, instance in enumerate(label["Instances"],1):
        #        print(
        #            f"  └── Instance {i}: "
        #            f"Confidence {instance['Confidence']:.1f}%"
        #        )

    print("-" * 50)

    # 2. Compare faces with reference library
    print("\nFacial recognition results:")
    print("-" * 50)
    face_comparison = analyzer.compare_faces_with_library(input_image, known_faces_dir)

    if face_comparison:
        if "error" in face_comparison:
            print(f"Unable to do facial recognition: {face_comparison['error']}")
        elif "matches" in face_comparison and face_comparison["matches"]:
            print("Known faces found in image:")
            for match in face_comparison["matches"]:
                person = match["person"]
                similarity = match["similarity"]
                # Create a visual confidence indicator
                confidence_bar = "█" * int(similarity / 10)  # Each block represents 10%
                print(f"\n  └── {person}")
                print(f"      Similarity: {similarity:.1f}% {confidence_bar}")
        else:
            print("No known faces were matched in this image.")
    else:
        print("No face comparison results available.")

    print("-" * 50)

    # 3. Get face details
    print("\nFace Details Results:")
    print("-" * 50)
    face_details = analyzer.detect_and_return_face_details(input_image)

    if face_details and "faces" in face_details:
        faces = face_details["faces"]
        print(f"Found {face_details['faces_found']} faces in the image")

        for face in faces:
            print(f"\nFace {face['face_number']}:")

            # Age Range
            print(
                f"  └── Age: {face['age_range']['Low']}-{face['age_range']['High']} years"
            )

            # Gender
            print(f"  └── Gender: {face['gender']}")

            # Primary Emotion
            confidence_bar = "█" * int(face["emotion_confidence"] / 10)
            print(f"  └── Emotion: {face['primary_emotion']}")
            print(
                f"      Confidence: {face['emotion_confidence']:.1f}% {confidence_bar}"
            )

            # Additional characteristics with confidence bars
            print("  └── Characteristics:")
            characteristics = {
                "eyeglasses": "Wearing glasses",
                "sunglasses": "Wearing sunglasses",
                "beard": "Has beard",
                "mustache": "Has mustache",
                "eyes_open": "Eyes open",
                "mouth_open": "Mouth open",
                "smile": "Smiling",
                "face_occluded": "Face occluded",
            }

            for key, label in characteristics.items():
                char = face["characteristics"][key]
                if char["value"]:  # Only show if True
                    confidence_bar = "█" * int(char["confidence"] / 10)
                    print(
                        f"      └── {label}: {char['confidence']:.1f}% {confidence_bar}"
                    )
    else:
        print("No faces detected in the image.")

    print("-" * 50)

    # 4. Draw boxes and save image
    # Pass both input_image and known_faces_dir to draw_bounding_boxes
    marked_image = analyzer.draw_bounding_boxes(input_image, known_faces_dir)
    marked_image.save(output_path)
    print(f"Analyzed image saved to: {output_path}")

    print("\n=== Analysis Complete ===")

    # Speech
    intro = "Greetings. I am Brutus. I am an artificial intelligence robot created by my Master Andy. At the moment I am able to do image recognition and produce audio. In future I will be able to do much more."
    # whynolegs = "My master foresaw a bleak future and in his wisdom decided to take my legs off so I would not kill him in his sleep."

    audio_folder = "audio"
    audio_filename = "intro.mp3"

    audio_output_file = os.path.join(audio_folder, audio_filename)

    speech_generator.text_to_speech(intro, audio_output_file, speech_rate=85)


if __name__ == "__main__":
    main()
