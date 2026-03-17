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
        if ext == '.bmp': return 'image/bmp'
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
        总结特定类别的多张图像特征。
        """
        print(f"正在学习类别: {category_name}...")
        content = [{"type": "text", "text": f"这些是'{category_name}'类别的多张图片。请详细总结该类别的共同视觉特征和定义性特征。该总结将作为未来图像分类的参考经验值。请用中文回答。"}]

        # 限制图像数量以避免超出上下文窗口
        selected_images = image_paths[:max_images]
        if len(image_paths) > max_images:
            print(f"注意：仅使用前 {max_images} 张图片进行训练。")

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
                print(f"跳过无法编码的图像: {path}")

        if len(content) == 1:
            return "没有提供有效的图像进行学习。"

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
        """将学习到的总结（经验值）保存到本地 JSON 文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.category_summaries, f, ensure_ascii=False, indent=4)
        print(f"经验值已保存至 {filepath}")

    def load_experience(self, filepath="experience.json"):
        """从本地 JSON 文件加载学习到的总结。"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.category_summaries = json.load(f)
            print(f"经验值已从 {filepath} 加载")
        else:
            print(f"未找到文件: {filepath}")

    def classify_image(self, image_path):
        """
        根据学习到的类别总结对新图像进行分类。
        """
        if not self.category_summaries:
            return "尚未学习任何类别。请先运行 learn_category 或加载经验值。"

        print(f"正在对图像进行分类: {image_path}...")
        data_url = self._encode_image(image_path)
        if not data_url:
            return "图像编码失败。"

        # 构建学习到的知识上下文
        knowledge_context = ""
        for name, summary in self.category_summaries.items():
            knowledge_context += f"--- 类别: {name} ---\n特征总结: {summary}\n\n"

        prompt = f"""
你是一位专业的图像分类专家。基于以下各类别已学习到的“经验值”（特征总结）：

{knowledge_context}

请分析提供的图像，并判断它属于哪个类别。
如果它匹配其中一个类别，请输出该【类别名称】。
如果它不匹配任何已知类别，请输出“未知”。
同时请简要说明你的判断理由。请用中文回答。
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
