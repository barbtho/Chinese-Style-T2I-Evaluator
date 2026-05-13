# evaluate2.py
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline  # ← 改为 SD
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
import gc
from translate import Translator

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def translate_chinese_to_english(text):
    """将中文提示词翻译成英文"""
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if not has_chinese:
        return text
    try:
        translator = Translator(to_lang="en", from_lang="zh")
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"翻译失败: {e}")
        return text

class TextToImageEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"使用设备: {device}")
        torch.cuda.empty_cache()
        gc.collect()

        # 初始化 Stable Diffusion 2.1 模型（替代 PixArt）
        print("正在加载 Stable Diffusion 2.1 模型...")
        self.text_to_image_pipe = StableDiffusionPipeline.from_pretrained(
            "D:/text2image/stable-diffusion-2-1-base",
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,  # ← 必须
            safety_checker=None,
        ).to(device)

        # 启用内存优化（SD 支持）
        self.text_to_image_pipe.enable_attention_slicing()
        try:
            self.text_to_image_pipe.vae.enable_tiling()
            print("✅ VAE分块解码已启用")
        except:
            print("⚠️ VAE分块解码不可用")

        # ⚠️ 删除 CPU offload（在 8GB 显存下反而有害）
        # try:
        #     self.text_to_image_pipe.enable_sequential_cpu_offload()
        #     print("✅ CPU卸载已启用")
        # except:
        #     print("⚠️ CPU卸载不可用")

        torch.cuda.empty_cache()
        gc.collect()

        # 初始化 BLIP2 图生文模型（保持不变）
        print("正在加载 BLIP2 模型...")
        self.processor = Blip2Processor.from_pretrained(
            "D:/text2image/BLIP2",
            local_files_only=True,
            use_fast=True  # ← 加速 tokenizer
        )
        self.image_to_text_model = Blip2ForConditionalGeneration.from_pretrained(
            "D:/text2image/BLIP2",
            torch_dtype=torch.float32,  # ← 改为 float32
            local_files_only=True,
            device_map="cpu"
        )

        # 初始化 Sentence-BERT（保持不变）
        print("正在加载 Sentence-BERT 模型...")
        self.sentence_model = SentenceTransformer('D:/text2image/sentence-bert')

        # 初始化 CLIP（保持不变）
        print("正在加载 CLIP 模型...")
        self.clip_model = SentenceTransformer('D:/text2image/clip')

        print("所有模型加载完成!")

    def generate_image_from_text(self, prompt, width=512, height=512, num_inference_steps=10, guidance_scale=7.5):
        """使用 Stable Diffusion 从文本生成图像"""
        # SD 2.1 原生支持 512x512，不建议用 1024
        if width != 512 or height != 512:
            print("⚠️ Stable Diffusion 2.1 仅推荐使用 512x512 分辨率")
            width, height = 512, 512

        english_prompt = translate_chinese_to_english(prompt)
        torch.cuda.empty_cache()
        gc.collect()

        image = self.text_to_image_pipe(
            english_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        torch.cuda.empty_cache()
        gc.collect()
        return image, english_prompt

    def generate_text_from_image(self, image):
        """使用 BLIP2 从图像生成文本描述（保持不变）"""
        torch.cuda.empty_cache()
        gc.collect()
        inputs = self.processor(images=image, return_tensors="pt")
        device = next(self.image_to_text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated_ids = self.image_to_text_model.generate(
            **inputs,
            max_length=50,
            num_beams=3,
            early_stopping=True
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        torch.cuda.empty_cache()
        gc.collect()
        return generated_text

    def calculate_semantic_similarity(self, text1, text2):
        """计算语义相似度（保持不变）"""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def calculate_clip_score(self, image, text):
        """计算 CLIP 分数（保持不变）"""
        image_embedding = self.clip_model.encode(image, convert_to_tensor=True)
        text_embedding = self.clip_model.encode(text, convert_to_tensor=True)
        clip_score = torch.nn.functional.cosine_similarity(image_embedding, text_embedding, dim=0)
        return float(clip_score)

    def evaluate_single_prompt(self, prompt, width=512, height=512, num_inference_steps=10, guidance_scale=7.5):
        """完整评估流程（保持不变）"""
        print(f"开始评估提示: '{prompt}'")
        generated_image, english_prompt = self.generate_image_from_text(
            prompt, width, height, num_inference_steps, guidance_scale
        )
        generated_text = self.generate_text_from_image(generated_image)
        semantic_similarity = self.calculate_semantic_similarity(english_prompt, generated_text)
        clip_score = self.calculate_clip_score(generated_image, english_prompt)
        results = {
            "original_text": prompt,
            "english_text": english_prompt,
            "generated_image": generated_image,
            "generated_text": generated_text,
            "semantic_similarity": semantic_similarity,
            "clip_score": clip_score
        }
        return results