# -*- coding: utf-8 -*-
# config.py
import os
import torch
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """
    系统全局配置类
    用于管理所有模型路径、硬件参数及API接口配置。
    """

    # --- 硬件与环境配置 ---
    # 自动检测计算设备
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 根据设备选择精度，节省显存
    TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    # --- 本地模型路径配置 ---
    # Stable Diffusion 模型路径
    PATH_STABLE_DIFFUSION: str = "D:/text2image/stable-diffusion-2-1-base"

    # CLIP 模型路径
    PATH_CLIP_MODEL: str = "D:/text2image/clip"

    # BLIP/Ollama 模型缓存路径 (如果有)
    PATH_MODEL_CACHE: str = "D:/text2image/cache"

    # --- Ollama 服务配置 ---
    # API 服务地址
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"

    # 视觉模型名称 (建议使用 minicpm-v)
    MODEL_NAME_VISION: str = "minicpm-v"

    # 裁判模型名称 (建议使用 qwen3:8b)
    MODEL_NAME_JUDGE: str = "qwen3:8b"

    # --- 生成参数默认值 ---
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    DEFAULT_STEPS: int = 20
    DEFAULT_GUIDANCE_SCALE: float = 7.5

    # --- 系统运行参数 ---
    # 是否开启详细日志
    VERBOSE_LOGGING: bool = True

    # 显存优化级别 (0:无, 1:CPU卸载, 2:激进卸载)
    VRAM_OPTIMIZATION_LEVEL: int = 2

    @staticmethod
    def validate_paths() -> bool:
        """
        校验关键路径是否存在。

        Returns:
            bool: 如果所有路径有效返回 True，否则 False
        """
        paths = [
            SystemConfig.PATH_STABLE_DIFFUSION,
            SystemConfig.PATH_CLIP_MODEL
        ]

        for p in paths:
            if not os.path.exists(p):
                print(f"配置文件警告: 路径不存在 -> {p}")
                return False
        return True


# 实例化配置对象供外部调用
cfg = SystemConfig()