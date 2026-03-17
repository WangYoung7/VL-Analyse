import os
import argparse
from vl_classifier import VLClassifier

def list_image_files(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(valid_extensions)]

# --- 配置区 ---
STORAGE_DIR = "vl_experience"  # 经验值固定存放文件夹
SAVE_ENABLED = 1               # 是否保存经验值 (1: 开启, 0: 关闭)
# --------------

def main():
    parser = argparse.ArgumentParser(description="硅基流动 (SiliconFlow) VL 图像分类交互工具")
    parser.add_argument("--api_key", help="硅基流动 API Key (或设置 SILICONFLOW_API_KEY 环境变量)")
    parser.add_argument("--dataset", help="包含类别子文件夹的数据集目录路径")
    args = parser.parse_args()

    try:
        classifier = VLClassifier(
            api_key=args.api_key,
            storage_dir=STORAGE_DIR,
            save_enabled=SAVE_ENABLED
        )
        # 启动时自动加载已有经验
        classifier.load_experience()
    except ValueError as e:
        print(f"错误: {e}")
        return

    while True:
        print("\n" + "="*40)
        print("   硅基流动 VL 图像分类交互工具")
        print("="*40)
        print("1. 批量学习数据集特征 (子文件夹形式)")
        print("2. 设定/修改人为经验值")
        print("3. 生成类别差异分析 (推荐)")
        print("4. 判定新图像类别")
        print("5. 查看当前所有经验值")
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
            if not classifier.experience_data:
                print("尚未加载或学习任何类别。请通过选项1学习类别。")
                continue
            print("\n当前类别列表:")
            cats = list(classifier.experience_data.keys())
            for i, cat in enumerate(cats):
                print(f"{i+1}. {cat}")

            try:
                idx = int(input("请选择要更新人为经验的类别编号: ")) - 1
                if 0 <= idx < len(cats):
                    cat_name = cats[idx]
                    manual_text = input(f"请输入对 '{cat_name}' 的人为经验描述: ").strip()
                    classifier.update_manual_experience(cat_name, manual_text)
                else:
                    print("编号无效。")
            except ValueError:
                print("请输入有效的数字。")

        elif choice == '3':
            result = classifier.analyze_category_differences()
            print(f"\n类别间区别分析:\n{result}")

        elif choice == '4':
            img_path = input("请输入要分类的图像路径: ").strip()
            if os.path.exists(img_path):
                result = classifier.classify_image(img_path)
                print(f"\n判定结果:\n{result}")
            else:
                print("找不到该图像文件。")

        elif choice == '5':
            if classifier.experience_data:
                print("\n" + "-"*30)
                print(f"存储目录: {STORAGE_DIR} | 自动保存: {'开启' if SAVE_ENABLED else '关闭'}")
                print("-"*30)
                for cat, data in classifier.experience_data.items():
                    print(f"【类别】: {cat}")
                    print(f"  [总结经验]: {data['summary'][:100]}..." if data['summary'] else "  [总结经验]: (空)")
                    print(f"  [人为经验]: {data['manual']}" if data['manual'] else "  [人为经验]: (空)")
                if classifier.differences_analysis:
                    print(f"\n【类别间区别分析】:\n{classifier.differences_analysis[:200]}...")
                print("-"*30)
            else:
                print("当前没有任何经验数据。")

        elif choice == 'q':
            break
        else:
            print("无效的选择。")

if __name__ == "__main__":
    main()
