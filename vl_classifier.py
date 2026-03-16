import base64
import json
import os
from openai import OpenAI

class VLClassifier:
    def __init__(self, api_key=None, base_url="https://api.siliconflow.cn/v1", model="Qwen/Qwen2-VL-72B-Instruct"):
        # If api_key is not provided, try to get it from environment variable
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Please provide it or set SILICONFLOW_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.category_summaries = {}

    def _get_mime_type(self, image_path):
        """Simple mime type detection based on extension."""
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png': return 'image/png'
        if ext in ['.jpg', '.jpeg']: return 'image/jpeg'
        if ext == '.webp': return 'image/webp'
        return 'image/jpeg' # fallback

    def _encode_image(self, image_path):
        """Encodes an image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = self._get_mime_type(image_path)
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def learn_category(self, category_name, image_paths, max_images=10):
        """
        Summarizes features of multiple images for a specific category.
        """
        print(f"Learning category: {category_name}...")
        content = [{"type": "text", "text": f"These are multiple images of '{category_name}'. Please summarize the common visual characteristics and defining features of this category in detail. This summary will be used as reference for future classification."}]

        # Limit the number of images to avoid context window issues
        selected_images = image_paths[:max_images]
        if len(image_paths) > max_images:
            print(f"Notice: Only using the first {max_images} images for training.")

        for path in selected_images:
            data_url = self._encode_image(path)
            if data_url:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                })
            else:
                print(f"Skipping {path} due to encoding error.")

        if len(content) == 1:
            return "No valid images provided for learning."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": content}
            ]
        )

        summary = response.choices[0].message.content
        self.category_summaries[category_name] = summary
        print(f"Summary for {category_name} completed.")
        return summary

    def save_experience(self, filepath="experience.json"):
        """Saves learned summaries (experience values) to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.category_summaries, f, ensure_ascii=False, indent=4)
        print(f"Experience values saved to {filepath}")

    def load_experience(self, filepath="experience.json"):
        """Loads learned summaries from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.category_summaries = json.load(f)
            print(f"Experience values loaded from {filepath}")
        else:
            print(f"File {filepath} not found.")

    def classify_image(self, image_path):
        """
        Classifies a new image based on learned category summaries.
        """
        if not self.category_summaries:
            return "No categories learned yet. Please run learn_category first or load experience."

        print(f"Classifying image: {image_path}...")
        data_url = self._encode_image(image_path)
        if not data_url:
            return "Failed to encode image."

        # Build the learned knowledge context
        knowledge_context = ""
        for name, summary in self.category_summaries.items():
            knowledge_context += f"--- Category: {name} ---\nCharacteristics: {summary}\n\n"

        prompt = f"""
You are an expert image classifier. Based on the following 'experience values' (learned characteristics) of different categories:

{knowledge_context}

Please analyze the provided image and determine which category it belongs to.
If it matches one of the categories, output the Category Name.
If it doesn't match any, output 'Unknown'.
Also provide a brief explanation for your judgment.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during classification: {e}"

if __name__ == "__main__":
    # Example usage:
    # 1. Set your API Key: export SILICONFLOW_API_KEY='sk-xxxxx'
    # 2. Or pass it directly: classifier = VLClassifier(api_key="sk-xxxxx")

    try:
        # For demonstration, we use a placeholder key. User should replace it.
        api_key = os.getenv("SILICONFLOW_API_KEY", "sk-xxxxx")
        classifier = VLClassifier(api_key=api_key)

        print("-" * 30)
        print("VL Classifier for SiliconFlow")
        print("-" * 30)
        print("Usage Example (Manual Steps):")
        print("1. classifier.learn_category('Apple', ['apple1.jpg', 'apple2.jpg'])")
        print("2. classifier.learn_category('Banana', ['banana1.jpg', 'banana2.jpg'])")
        print("3. result = classifier.classify_image('unknown_fruit.jpg')")
        print("4. classifier.save_experience() # To reuse summaries later")
        print("-" * 30)

    except ValueError as e:
        print(f"Initialization Error: {e}")
