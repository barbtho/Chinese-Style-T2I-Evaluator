# -*- coding: utf-8 -*-
# evaluate_advanced.py

"""
TextToImageEvaluator Core Module
--------------------------------
本模块实现了文生图模型评测的核心后端逻辑。
集成 Stable Diffusion 生成、MiniCPM-V 视觉理解、CLIP 客观评分及 Qwen 语义裁判。

Author: System
Date: 2024-12
License: Proprietary
"""

import os
import torch
import gc
import json
import requests
import base64
import re
import warnings
from io import BytesIO
from typing import Dict, Any, Tuple, Optional, Union, List

# --- 第三方深度学习库导入 ---
from diffusers import StableDiffusionPipeline
from sentence_transformers import SentenceTransformer
from translate import Translator

# --- 尝试导入本地配置 (软著工程化结构) ---
try:
    from config import cfg
    from logger_setup import sys_logger
except ImportError:
    pass

# 忽略不必要的警告，保持控制台清洁
warnings.filterwarnings("ignore")


class TextToImageEvaluator:
    """
    智能文生图模型评测核心控制器类。

    采用 Time-Sharing (分时复用) 显存管理策略，
    专为消费级显卡 (8GB VRAM) 优化，确保多模型串行运行不溢出。
    """

    def __init__(
            self,
            device: Optional[str] = None
    ) -> None:
        """
        初始化评测器实例，加载各个子模型。

        Args:
            device (str, optional): 指定计算设备 ('cuda' 或 'cpu')。
                                    若未指定，将自动检测系统硬件。
        """
        # 1. 硬件设备检测
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] 🖥️  检测到计算设备: {self.device}")
        print(f"[{self.__class__.__name__}] 🚀 正在启动 8GB 显存优化模式 (Time-Sharing)...")

        # 2. 初始化各个子模块
        self._init_translator()
        self._init_stable_diffusion()
        self._init_clip_model()

        print(f"[{self.__class__.__name__}] ✅ 系统核心加载完成！显存空闲中，准备就绪。")

    def _init_translator(self) -> None:
        """
        [内部方法] 初始化中英翻译模块。
        用于将用户的中文 Prompt 转换为 SD 模型所需的英文 Prompt。
        """
        try:
            self.translator = Translator(to_lang="en", from_lang="zh")
            print("   ├── ✅ 翻译模块初始化成功")
        except Exception as e:
            self.translator = None
            print(f"   ├── ⚠️ 翻译模块初始化失败 (将使用原文): {e}")

    def _init_stable_diffusion(self) -> None:
        """
        [内部方法] 初始化 Stable Diffusion 文生图模型。

        策略：
        - 初始化时加载到 CPU 内存 (RAM)。
        - 仅在生成阶段移动到 GPU 显存 (VRAM)。
        - 启用 fp16 半精度和 attention slicing 优化。
        """
        print("   ├── 📥 正在加载 Stable Diffusion 2.1 (CPU待机模式)...")

        model_path = "D:/text2image/stable-diffusion-2-1-base"

        try:
            self.text_to_image_pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
                safety_checker=None
            ).to("cpu")

            # 启用显存优化
            self.text_to_image_pipe.enable_attention_slicing()

            # 尝试启用 VAE Tiling (大幅降低高分辨率生成的显存占用)
            try:
                self.text_to_image_pipe.vae.enable_tiling()
                print("   ├── ✅ VAE Tiling 显存优化已启用")
            except AttributeError:
                pass

        except Exception as e:
            print(f"   ├── ❌ Stable Diffusion 加载失败: {e}")
            raise e

    def _init_clip_model(self) -> None:
        """
        [内部方法] 初始化 CLIP 模型。
        用于计算图像与文本的客观余弦相似度。
        """
        print("   ├── 📥 正在加载 CLIP 模型 (用于客观评分)...")
        clip_path = "D:/text2image/clip"

        try:
            self.clip_model = SentenceTransformer(clip_path, device="cpu")
        except Exception as e:
            print(f"   ├── ❌ CLIP 模型加载失败: {e}")
            self.clip_model = None

    def flush_vram(self) -> None:
        """
        显存清理大师。
        强制调用 Python 垃圾回收器和 PyTorch 显存缓存清理。
        应在模型切换（如 SD -> Ollama）之间调用。
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def translate_text(self, text: str) -> str:
        """
        辅助工具：中文文本转英文。

        Args:
            text (str): 输入的原始文本 (可能是中文)

        Returns:
            str: 翻译后的英文文本。如果输入不含中文或翻译失败，返回原文。
        """
        # 简单判断是否包含中文字符范围
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

        if not has_chinese:
            return text

        try:
            if self.translator:
                return self.translator.translate(text)
        except Exception:
            # 发生异常时静默失败，返回原文
            pass

        return text

    def generate_image_from_text(
            self,
            prompt: str,
            width: int = 512,
            height: int = 512,
            steps: int = 20,
            scale: float = 7.5
    ) -> Tuple[Any, str]:
        """
        🎨 生成阶段：执行 Stable Diffusion 生成任务。

        包含完整的显存调度生命周期：
        1. 翻译 Prompt
        2. 清理显存
        3. 模型上机 (CPU -> GPU)
        4. 推理生成
        5. 模型下机 (GPU -> CPU)
        6. 再次清理

        Args:
            prompt (str): 用户输入的提示词
            width (int): 图像宽度
            height (int): 图像高度
            steps (int): 推理步数
            scale (float): CFG 引导系数

        Returns:
            Tuple[Any, str]: (生成的 PIL 图像对象, 英文提示词)
        """
        print("🎨 [1/4] 正在唤醒 SD 模型到显卡...")

        # 1. 预处理提示词
        english_prompt = self.translate_text(prompt)

        # 2. 显存调度：准备上线
        self.flush_vram()
        self.text_to_image_pipe.to(self.device)

        # 3. 执行推理
        print(f"   ├── 正在绘制: {prompt[:20]}...")
        try:
            image = self.text_to_image_pipe(
                english_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=scale
            ).images[0]
        except Exception as e:
            print(f"   ├── ❌ 生成失败: {e}")
            # 生成失败时返回一个空白图防止程序崩溃
            from PIL import Image
            image = Image.new('RGB', (width, height), color='black')
        finally:
            # 4. 显存调度：立即下线 (为 Ollama 让路)
            print("   ├── 绘制完毕，SD 模型正在休眠...")
            self.text_to_image_pipe.to("cpu")
            self.flush_vram()

        return image, english_prompt

    def generate_text_from_image(
            self,
            image: Any
    ) -> str:
        """
        👁️ 视觉阶段：调用 Ollama (MiniCPM-V) 进行图像理解。

        目标：生成高质量的中文描述，精准识别国风元素 (如水墨、汉服、飞檐等)。

        Args:
            image (Any): PIL 格式的输入图像

        Returns:
            str: 视觉大模型生成的中文描述文本
        """
        print("👁️ [2/4] 正在调用视觉模型 (MiniCPM-V)...")

        # 1. 图像转码 Base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 2. 构造视觉提示词
        vision_prompt = (
            "请用中文极其详细地描述这张图片。\n"
            "1. 重点关注艺术风格（如：水墨画、工笔画、赛博朋克、写实摄影等）。\n"
            "2. 详细描述画面中的物体、人物衣着（如汉服）、动作和表情。\n"
            "3. 描述光影、色彩运用以及画面的整体意境。\n"
            "4. 如果画面中有中国传统元素，请准确识别并描述。"
        )

        model_name = "minicpm-v"

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': vision_prompt,
                    'images': [img_str],
                    'stream': False,
                    # keep_alive: 0 表示用完立刻卸载模型，不占显存
                    'keep_alive': 0,
                    'options': {
                        'temperature': 0.2,
                        'num_predict': 400
                    }
                },
                timeout=180
            )

            if response.status_code == 200:
                res = response.json().get('response', '')
                print(f"   ├── 视觉描述完成 (前30字): {res[:30]}...")
                return res
            else:
                error_msg = f"❌ 视觉模型 API 错误: {response.status_code}"
                print(f"   ├── {error_msg}")
                return error_msg

        except Exception as e:
            print(f"   ├── ⚠️ 视觉识别网络异常: {e}")
            return "视觉识别服务连接失败"

    def calculate_clip_score(
            self,
            image: Any,
            text: str
    ) -> float:
        """
        📎 CLIP 阶段：计算客观图文匹配度。

        Args:
            image (Any): PIL 图像
            text (str): 英文提示词

        Returns:
            float: 余弦相似度分数 (0.0 - 1.0)
        """
        print("📎 [3/4] 计算 CLIP 分数 (CPU模式)...")

        if self.clip_model is None:
            return 0.0

        try:
            # 确保在 CPU 上计算，避免占用显存
            image_embedding = self.clip_model.encode(
                image,
                convert_to_tensor=True,
                device="cpu"
            )
            text_embedding = self.clip_model.encode(
                text,
                convert_to_tensor=True,
                device="cpu"
            )

            # 计算余弦相似度
            score = torch.nn.functional.cosine_similarity(
                image_embedding,
                text_embedding,
                dim=0
            )
            return float(score)

        except Exception as e:
            print(f"   ├── ⚠️ CLIP 计算异常: {e}")
            return 0.0

    def _construct_judge_prompt(
            self,
            original_req: str,
            actual_img_desc: str
    ) -> str:
        """
        [内部方法] 构造 AI 裁判的 Prompt。
        这是解决“分数固定”和“变量名泄露”问题的核心代码。
        """
        return f"""
        你是一位精通中国传统文化与现代艺术的图像评测专家。
        请对比以下两段中文描述，判断生成图像与原始需求的符合程度。

        【原始需求】: "{original_req}"
        【生成图实际画面】: "{actual_img_desc}"

        请完成两项任务：

        任务一：深度打分 (百分制)
        请从以下维度打分（⚠️请务必根据实际情况计算，范围 0-100，禁止抄袭示例数值）：
        - object_score (实体对象)
        - style_score (风格意境)
        - quantity_score (数量关系)
        - spatial_score (空间方位)
        - color_score (色彩准确)
        - overall_semantic_score (语义一致)
        - total_score (综合评分)

        任务二：撰写评语
        撰写一段流畅的中文评语 (reasoning)，严禁在评语中出现 "object_score" 等变量名。

        【输出格式要求】：
        请严格按照以下 JSON 结构输出（示例中的数值 "0" 仅为占位符，请填入真实分数）：
        {{
            "object_score": 0,
            "style_score": 0,
            "quantity_score": 0,
            "spatial_score": 0,
            "color_score": 0,
            "overall_semantic_score": 0,
            "total_score": 0,  
            "reasoning": "这里填写你的真实分析..."
        }}
        """

    def calculate_ai_similarity_score(
            self,
            prompt1: str,
            prompt2: str
    ) -> Dict[str, Union[int, str]]:
        """
        🤖 裁判阶段：调用 LLM (Qwen) 进行全中文评测。

        包含以下核心逻辑：
        1. 构造防污染 Prompt。
        2. 调用 Ollama API。
        3. 解析 JSON。
        4. **Score Normalization**: 强制将 0-1 的小数转换为 0-100 的整数。

        Args:
            prompt1 (str): 原始中文需求
            prompt2 (str): 视觉模型生成的中文描述

        Returns:
            Dict: 包含各维度评分(整数)和评语的字典
        """
        print("🤖 [4/4] AI 裁判打分中 (全中文模式)...")

        # 定义保底数据结构
        fallback_data = {
            "object_score": 0,
            "style_score": 0,
            "quantity_score": 0,
            "spatial_score": 0,
            "color_score": 0,
            "overall_semantic_score": 0,
            "total_score": 0,
            "reasoning": "评分系统连接超时或返回错误。"
        }

        # 构造提示词
        eval_prompt = self._construct_judge_prompt(prompt1, prompt2)

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': "qwen3:8b",
                    'prompt': eval_prompt,
                    'format': 'json',
                    'stream': False,
                    'keep_alive': 0
                },
                timeout=180
            )

            if response.status_code == 200:
                content = response.json().get('response', '{}')

                # 清洗可能存在的 Markdown 标记
                if "```" in content:
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        content = match.group(0)

                data = json.loads(content)

                # =========== 🔥 核心数据清洗逻辑 🔥 ===========
                # 遍历所有分数字段，执行归一化处理
                score_keys = [
                    "object_score", "style_score", "quantity_score",
                    "spatial_score", "color_score", "overall_semantic_score",
                    "total_score"
                ]

                for key in score_keys:
                    val = data.get(key, 0)

                    # 逻辑：如果分数在 0~1 之间 (如 0.85)，说明模型打成了小数 -> 乘100
                    # 如果分数 > 1 (如 85)，说明模型打对了 -> 保持不变
                    if 0 < val <= 1.0:
                        data[key] = int(val * 100)
                    else:
                        data[key] = int(val)
                # ===============================================

                # 补全缺失字段
                for k, v in fallback_data.items():
                    if k not in data:
                        data[k] = v

                return data

        except Exception as e:
            print(f"   ├── ⚠️ 评分服务异常: {e}")
            fallback_data['reasoning'] = f"系统报错: {str(e)}"
            return fallback_data

        return fallback_data

    def evaluate_single_prompt(
            self,
            prompt: str,
            width: int = 512,
            height: int = 512,
            steps: int = 20,
            scale: float = 7.5
    ) -> Dict[str, Any]:
        """
        评测系统主入口函数 (Facade Pattern)。

        串联整个评测流水线：
        Generate -> Vision -> CLIP -> Judge -> Result Package

        Args:
            prompt (str): 提示词
            width (int): 宽
            height (int): 高
            steps (int): 步数
            scale (float): 引导系数

        Returns:
            Dict[str, Any]: 包含所有中间结果和最终评分的完整数据包
        """
        # 1. 生成图
        image, english_prompt = self.generate_image_from_text(
            prompt, width, height, steps, scale
        )

        # 2. 视觉理解
        generated_chinese_text = self.generate_text_from_image(image)

        # 3. CLIP 客观评分
        clip_score = self.calculate_clip_score(image, english_prompt)

        # 4. AI 主观裁判
        ai_result = self.calculate_ai_similarity_score(
            prompt, generated_chinese_text
        )

        # 5. 打包结果
        result_packet = {
            "original_text": prompt,
            "english_text": english_prompt,
            "generated_image": image,
            "generated_text": generated_chinese_text,
            "clip_score": clip_score,
            "ai_similarity": ai_result
        }

        return result_packet