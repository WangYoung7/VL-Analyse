import os
import argparse
from vl_classifier import VLClassifier

def list_image_files(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(valid_extensions)]

def main():
    parser = argparse.ArgumentParser(description="硅基流动 (SiliconFlow) VL 图像分类交互工具")
    parser.add_argument("--api_key", help="硅基流动 API Key (或设置 SILICONFLOW_API_KEY 环境变量)")
    parser.add_argument("--dataset", help="包含类别子文件夹的数据集目录路径")
    args = parser.parse_args()

    try:
        classifier = VLClassifier(api_key=args.api_key)
    except ValueError as e:
        print(f"错误: {e}")
        return

    while True:
        print("\n" + "="*40)
        print("   硅基流动 VL 图像分类交互工具")
        print("="*40)
        print("1. 从数据集文件夹学习特征")
        print("2. 对单张图像进行分类")
        print("3. 保存经验值到本地")
        print("4. 从本地加载经验值")
        print("5. 查看已学习的类别")
        print("q. 退出")

        choice = input("\n请选择操作: ").strip().lower()

        if choice == '1':
            dataset_path = args.dataset or input("请输入数据集目录路径: ").strip()
            if not os.path.isdir(dataset_path):
                print(f"错误: {dataset_path} 不是一个有效的目录。")
                continue

            # 子文件夹即为类别名称
            subfolders = sorted([f.path for f in os.scandir(dataset_path) if f.is_dir()])
            if not subfolders:
                print("在数据集目录中未找到子文件夹。")
                continue

            for folder in subfolders:
                category_name = os.path.basename(folder)
                images = list_image_files(folder)
                if images:
                    print(f"\n发现类别 '{category_name}' 的 {len(images)} 张图片")
                    classifier.learn_category(category_name, images)
                else:
                    print(f"在 {folder} 中未找到图片")

        elif choice == '2':
            img_path = input("请输入要分类的图像路径: ").strip()
            if os.path.exists(img_path):
                result = classifier.classify_image(img_path)
                print(f"\n分类结果:\n{result}")
            else:
                print("找不到该文件。")

        elif choice == '3':
            path = input("请输入保存文件名 (默认: experience.json): ").strip() or "experience.json"
            classifier.save_experience(path)

        elif choice == '4':
            path = input("请输入加载文件名 (默认: experience.json): ").strip() or "experience.json"
            classifier.load_experience(path)

        elif choice == '5':
            if classifier.category_summaries:
                print("\n已学习的类别:")
                for cat in classifier.category_summaries.keys():
                    print(f"- {cat}")
            else:
                print("尚未学习任何类别。")

        elif choice == 'q':
            break
        else:
            print("无效的选择。")

if __name__ == "__main__":
    main()
