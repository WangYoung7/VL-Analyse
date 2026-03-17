import base64
import json
import os
from openai import OpenAI

class VLClassifier:
    def __init__(self, api_key=None, base_url="https://api.siliconflow.cn/v1",
                 model="Qwen/Qwen2-VL-72B-Instruct", storage_dir="vl_experience", save_enabled=1):
        # 如果未提供 api_key，尝试从环境变量获取
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("需要 API Key。请提供它或设置 SILICONFLOW_API_KEY 环境变量。")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.storage_dir = storage_dir
        self.save_enabled = save_enabled  # 1 为保存，0 为不保存

        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        # 结构：{category_name: {"summary": str, "manual": str}}
        self.experience_data = {}
        self.differences_analysis = ""

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
        """总结特定类别的多张图像特征。"""
        print(f"正在学习类别: {category_name}...")
        content = [{"type": "text", "text": f"这些是'{category_name}'类别的多张图片。请详细总结该类别的共同视觉特征和定义性特征。该总结将作为未来图像分类的参考经验值。请用中文回答。"}]

        selected_images = image_paths[:max_images]
        for path in selected_images:
            data_url = self._encode_image(path)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})

        if len(content) == 1:
            return "没有提供有效的图像进行学习。"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}]
        )

        summary = response.choices[0].message.content
        if category_name not in self.experience_data:
            self.experience_data[category_name] = {"summary": "", "manual": ""}
        self.experience_data[category_name]["summary"] = summary
        print(f"{category_name} 的总结经验值已生成。")
        self._auto_save()
        return summary

    def update_manual_experience(self, category_name, manual_text):
        """更新人为经验值。"""
        if category_name not in self.experience_data:
            self.experience_data[category_name] = {"summary": "", "manual": ""}
        self.experience_data[category_name]["manual"] = manual_text
        print(f"{category_name} 的人为经验值已更新。")
        self._auto_save()

    def analyze_category_differences(self):
        """分析已学习类别之间的区别。"""
        if len(self.experience_data) < 2:
            return "至少需要两个类别才能进行区别分析。"

        print("正在分析各类别之间的区别...")
        context = ""
        for name, data in self.experience_data.items():
            context += f"类别: {name}\n特征总结: {data['summary']}\n人为经验: {data['manual']}\n\n"

        prompt = f"""
以下是已学习的多个类别的特征数据：

{context}

请对比这些类别，详细分析它们之间的主要区别、容易混淆的点以及各自的独特标识特征。
请用中文回答，并以清晰的条目形式列出。
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        self.differences_analysis = response.choices[0].message.content
        self._auto_save()
        return self.differences_analysis

    def _auto_save(self):
        """内部自动保存方法。"""
        if self.save_enabled == 1:
            self.save_experience()

    def save_experience(self):
        """将学习到的经验数据保存到本地存储目录。"""
        filepath = os.path.join(self.storage_dir, "experience_data.json")
        data_to_save = {
            "experience": self.experience_data,
            "analysis": self.differences_analysis
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"数据已自动保存至 {filepath}")

    def load_experience(self):
        """从本地存储目录加载经验数据。"""
        filepath = os.path.join(self.storage_dir, "experience_data.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "experience" in data:
                    self.experience_data = data["experience"]
                    self.differences_analysis = data.get("analysis", "")
                else:
                    self.experience_data = {k: {"summary": v, "manual": ""} for k, v in data.items()}
            print(f"已从 {filepath} 恢复经验数据")
        else:
            print("未发现本地经验数据，将开始全新学习。")

    def classify_image(self, image_path):
        """
        根据学习到的类别经验（总结+人为）对新图像进行分类。
        """
        if not self.experience_data:
            return "尚未学习任何类别。请先运行学习程序或加载经验值。"

        print(f"正在对图像进行分类: {image_path}...")
        data_url = self._encode_image(image_path)
        if not data_url:
            return "图像编码失败。"

        # 构建知识上下文
        knowledge_context = ""
        for name, data in self.experience_data.items():
            knowledge_context += f"--- 类别: {name} ---\n"
            knowledge_context += f"[总结经验值]: {data['summary']}\n"
            knowledge_context += f"[人为经验值]: {data['manual'] if data['manual'] else '无'}\n\n"

        if self.differences_analysis:
            knowledge_context += f"--- 类别间区别分析 ---\n{self.differences_analysis}\n\n"

        prompt = f"""
你是一位专业的图像分类专家。请基于以下提供的参考资料对图像进行判断。

【参考资料】
{knowledge_context}

【判定原则】
1. 请同时参考[总结经验值]和[人为经验值]。
2. 在进行最终决策时，[总结经验值]的影响权重占比50%，[人为经验值]的影响权重占比50%。
3. 如果提供了[类别间区别分析]，请务必结合该分析来排除混淆项。

【任务】
请分析提供的图像，判断它属于哪个类别。
- 如果匹配，输出【类别名称】。
- 如果不匹配任何已知类别，输出“未知”。
- 请详细说明判断理由，并指出[总结经验值]和[人为经验值]分别是如何影响你的判断的。

请用中文回答。
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
