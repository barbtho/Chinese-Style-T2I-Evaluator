# test_pixart_lcm.py
import torch
import gc
from diffusers import PixArtAlphaPipeline
import os


def test_pixart_lcm():
    print("🚀 开始测试 PixArt-LCM（快速版）...")

    # 指定模型下载路径
    model_path = "D:/text2image/PixArt-LCM"

    print(f"📁 模型将下载到: {model_path}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()

    try:
        print("📥 正在下载并加载 PixArt-LCM 模型...")
        print("⚠️ 首次下载需要时间，请耐心等待...")

        # 使用标准的 PixArtAlphaPipeline，但加载 LCM 版本
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",  # LCM快速版本
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=model_path
        ).to("cuda")

        # 启用内存优化
        pipe.enable_attention_slicing()
        print("✅ 模型加载成功！")

        # 测试生成（只需4步！）
        test_prompts = [
            "a beautiful landscape with mountains and lake",
            "a cute cat sitting on a sofa",
            "a modern living room with large windows"
        ]

        for i, prompt in enumerate(test_prompts):
            print(f"\n🎨 生成图像 {i + 1}/3: {prompt}")

            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=4,  # 只需4步！
                    guidance_scale=0.0,  # LCM不需要guidance
                    width=768,  # 8GB显存用768x768
                    height=768
                ).images[0]

            filename = f"pixart_lcm_test_{i + 1}.png"
            image.save(filename)
            print(f"✅ 图像保存: {filename}")

        print(f"\n🎉 PixArt-LCM 测试完成！")
        print(f"📁 模型已保存到: {model_path}")
        print("⚡ 生成速度应该很快（每张图约3-8秒）")

        return True

    except Exception as e:
        print(f"❌ PixArt-LCM 测试失败: {e}")
        print("\n🔄 尝试备选方案：SDXL-Lightning...")
        return test_sdxl_lightning()


def test_sdxl_lightning():
    """备选方案：SDXL-Lightning"""
    try:
        from diffusers import StableDiffusionXLPipeline

        print("📥 正在下载 SDXL-Lightning...")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "ByteDance/SDXL-Lightning-4step",
            torch_dtype=torch.float16,
            cache_dir="D:/text2image/SDXL-Lightning"
        ).to("cuda")

        pipe.enable_attention_slicing()
        print("✅ SDXL-Lightning 加载成功！")

        image = pipe(
            "a cute cat, detailed, 4k",
            num_inference_steps=4
        ).images[0]

        image.save("sdxl_lightning_test.png")
        print("✅ SDXL-Lightning 测试成功！")
        return True

    except Exception as e:
        print(f"❌ SDXL-Lightning 也失败了: {e}")
        return False


if __name__ == "__main__":
    success = test_pixart_lcm()
    if success:
        print("\n🎊 恭喜！你可以用这个快速模型替换原来的SD 2.1了")
    else:
        print("\n😞 所有快速模型都尝试失败了")