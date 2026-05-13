from sentence_transformers import SentenceTransformer
import numpy as np


# 超简短版本
def quick_test():
    """超简短测试"""
    try:
        model = SentenceTransformer("D:/text2image/sentence-bert")
        print("✅ 模型加载成功")

        # 测试相似度计算
        text1 = "A college student was carrying a schoolbag"
        text2 = "a painting of a man walking down the street with a backpack"

        emb1 = model.encode(text1)
        emb2 = model.encode(text2)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        print(f"文本1: '{text1}'")
        print(f"文本2: '{text2}'")
        print(f"相似度: {similarity:.4f}")
        print("✅ 测试完成")

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    quick_test()