import os
import argparse
from vl_classifier import VLClassifier

def list_image_files(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(valid_extensions)]

def main():
    parser = argparse.ArgumentParser(description="SiliconFlow VL Classifier Interactive Tool")
    parser.add_argument("--api_key", help="SiliconFlow API Key (or set SILICONFLOW_API_KEY env var)")
    parser.add_argument("--dataset", help="Path to dataset directory with category subfolders")
    args = parser.parse_args()

    try:
        classifier = VLClassifier(api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return

    while True:
        print("\n" + "="*40)
        print("   SiliconFlow VL Interactive Tool")
        print("="*40)
        print("1. Learn from Dataset Folder")
        print("2. Classify a Single Image")
        print("3. Save Experience Values")
        print("4. Load Experience Values")
        print("5. List Learned Categories")
        print("q. Exit")

        choice = input("\nChoose an option: ").strip().lower()

        if choice == '1':
            dataset_path = args.dataset or input("Enter dataset directory path: ").strip()
            if not os.path.isdir(dataset_path):
                print(f"Error: {dataset_path} is not a directory.")
                continue

            # Subfolders are categories
            subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
            if not subfolders:
                print("No subfolders found in the dataset directory.")
                continue

            for folder in subfolders:
                category_name = os.path.basename(folder)
                images = list_image_files(folder)
                if images:
                    print(f"\nFound {len(images)} images for category '{category_name}'")
                    classifier.learn_category(category_name, images)
                else:
                    print(f"No images found in {folder}")

        elif choice == '2':
            img_path = input("Enter path to the image to classify: ").strip()
            if os.path.exists(img_path):
                result = classifier.classify_image(img_path)
                print(f"\nResult:\n{result}")
            else:
                print("File not found.")

        elif choice == '3':
            path = input("Enter filename to save (default: experience.json): ").strip() or "experience.json"
            classifier.save_experience(path)

        elif choice == '4':
            path = input("Enter filename to load (default: experience.json): ").strip() or "experience.json"
            classifier.load_experience(path)

        elif choice == '5':
            if classifier.category_summaries:
                print("\nLearned Categories:")
                for cat in classifier.category_summaries.keys():
                    print(f"- {cat}")
            else:
                print("No categories learned yet.")

        elif choice == 'q':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
