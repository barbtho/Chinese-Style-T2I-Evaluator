import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import matplotlib
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_test_image(image_path):
    """加载测试图像"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return None

        # 打开图像
        image = Image.open(image_path)
        print(f"✅ 成功加载图像: {image_path}")
        print(f"   图像尺寸: {image.size}, 模式: {image.mode}")

        # 如果图像不是RGB，转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("   已将图像转换为RGB模式")

        return image
    except Exception as e:
        print(f"❌ 加载图像失败: {e}")
        return None


def test_blip2_model_with_custom_image(image_path):
    """使用自定义图像测试BLIP2模型"""

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 模型路径
    model_path = "D:/text2image/BLIP2"

    print("=" * 50)
    print("BLIP2模型测试 - 使用自定义图像")
    print("=" * 50)


    # 4. 加载测试图像
    test_image = load_test_image(image_path)
    if test_image is None:
        return False

    try:
        # 5. 尝试加载模型
        print("\n正在加载BLIP2模型和处理器...")
        processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(device)

        print("✅ 模型加载成功")

        # 6. 使用模型生成描述
        print("使用模型生成图像描述...")
        inputs = processor(images=test_image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"✅ 模型生成描述: '{generated_text}'")

        # 7. 显示测试结果（使用中文字体）
        display_results_with_chinese(test_image, generated_text, image_path)

        print("\n🎉 BLIP2模型测试完全成功！")
        print("模型已正确下载并可以正常工作")
        return True

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_results_with_chinese(test_image, generated_text, image_path):
    """使用中文字体显示测试结果"""
    # 创建图形
    plt.figure(figsize=(12, 6))

    # 左侧显示测试图像
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title(f"测试图像: {os.path.basename(image_path)}", fontproperties='SimHei', fontsize=14)
    plt.axis('off')

    # 右侧显示结果
    plt.subplot(1, 2, 2)
    # 设置背景色
    plt.gca().set_facecolor('#f5f5f5')

    # 显示标题和结果
    plt.text(0.05, 0.9, "BLIP2测试结果", fontsize=16, weight='bold',
             fontproperties='SimHei', transform=plt.gca().transAxes)

    plt.text(0.05, 0.8, f"图像文件: {os.path.basename(image_path)}", fontsize=12,
             fontproperties='SimHei', transform=plt.gca().transAxes)

    plt.text(0.05, 0.7, "生成描述:", fontsize=14,
             fontproperties='SimHei', transform=plt.gca().transAxes)

    plt.text(0.05, 0.5, f"{generated_text}", fontsize=12,
             transform=plt.gca().transAxes, wrap=True)

    plt.text(0.05, 0.2, "✅ 模型测试成功", fontsize=14, color='green',
             fontproperties='SimHei', transform=plt.gca().transAxes)

    # 设置坐标轴范围并隐藏坐标轴
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 设置默认图像路径
    default_image_path = "output_small.jpg"

    # 检查默认图像是否存在
    if not os.path.exists(default_image_path):
        print(f"默认图像文件 '{default_image_path}' 不存在")
        print("请提供图像文件路径")

        # 让用户输入图像路径
        image_path = input("请输入图像文件路径: ").strip()

        # 移除可能的引号
        image_path = image_path.strip('"\'')
    else:
        image_path = default_image_path
        print(f"使用默认图像: {image_path}")

    # 运行测试
    success = test_blip2_model_with_custom_image(image_path)

    if success:
        print("\n" + "=" * 50)
        print("BLIP2模型准备就绪，可以用于文生图评估系统")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("BLIP2模型测试失败，请检查模型文件或图像文件")
        print("=" * 50)


if __name__ == "__main__":
    main()