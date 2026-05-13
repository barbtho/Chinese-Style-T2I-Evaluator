import os
from sentence_transformers import SentenceTransformer


def comprehensive_clip_test():
    model_path = "D:/text2image/CLIP"

    print("=" * 50)
    print("CLIP模型完整性验证")
    print("=" * 50)

    # 1. 检查文件
    required_files = [
        "0_CLIPModel/config.json",
        "0_CLIPModel/model.safetensors",  # 或 pytorch_model.bin
        "0_CLIPModel/tokenizer_config.json",
        "config_sentence_transformers.json"
    ]

    print("\n1. 文件检查:")
    all_files_exist = True
    for file in required_files:
        full_path = os.path.join(model_path, file)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file} - 缺失")
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ 文件不完整，无法加载模型")
        return False

    # 2. 尝试加载模型
    print("\n2. 模型加载测试:")
    try:
        model = SentenceTransformer(model_path)
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return False

    # 3. 功能测试
    print("\n3. 功能测试:")
    try:
        texts = ["a cute cat", "a beautiful landscape", "scientific research"]
        embeddings = model.encode(texts)

        print(f"   ✅ 编码测试成功")
        print(f"      输入文本: {len(texts)} 个")
        print(f"      输出维度: {embeddings.shape}")
        print(f"      向量类型: {type(embeddings)}")

        # 计算相似度
        similarity = embeddings[0] @ embeddings[1].T
        print(f"   ✅ 相似度计算: {similarity:.4f}")

        return True

    except Exception as e:
        print(f"   ❌ 功能测试失败: {e}")
        return False


if __name__ == "__main__":
    success = comprehensive_clip_test()
    print("\n" + "=" * 50)
    if success:
        print("🎉 CLIP模型验证成功！可以正常使用。")
    else:
        print("💥 CLIP模型验证失败！请检查下载。")
    print("=" * 50)