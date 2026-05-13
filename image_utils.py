# -*- coding: utf-8 -*-
# image_utils.py
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from typing import Optional, Union, Tuple


class ImageProcessor:
    """
    图像处理工具类
    包含图像格式转换、Base64编码解码、张量转换等静态方法。
    """

    @staticmethod
    def pil_to_base64(
            image: Image.Image,
            format: str = "JPEG",
            quality: int = 95
    ) -> str:
        """
        将 PIL Image 对象转换为 Base64 编码字符串。
        用于 API 传输。

        Args:
            image (Image.Image): 输入的 PIL 图像对象
            format (str, optional): 图像保存格式. Defaults to "JPEG".
            quality (int, optional): 压缩质量 (1-100). Defaults to 95.

        Returns:
            str: UTF-8 编码的 Base64 字符串
        """
        buffered = BytesIO()
        try:
            # 转换模式以兼容 JPEG
            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(buffered, format=format, quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        except Exception as e:
            print(f"图像转Base64失败: {e}")
            return ""

    @staticmethod
    def base64_to_pil(base64_str: str) -> Optional[Image.Image]:
        """
        将 Base64 字符串解码为 PIL Image 对象。

        Args:
            base64_str (str): Base64 编码字符串

        Returns:
            Optional[Image.Image]: 解码后的图像对象，失败返回 None
        """
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            print(f"Base64转图像失败: {e}")
            return None

    @staticmethod
    def normalize_image_for_clip(
            image: Image.Image,
            target_size: Tuple[int, int] = (224, 224)
    ) -> Image.Image:
        """
        标准化图像以适应 CLIP 模型输入。

        Args:
            image (Image.Image): 原始图像
            target_size (Tuple[int, int], optional): 目标分辨率. Defaults to (224, 224).

        Returns:
            Image.Image: 处理后的图像
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def save_temp_image(
            image: Image.Image,
            prefix: str = "temp",
            dir_path: str = "./outputs"
    ) -> str:
        """
        保存临时图像到磁盘。

        Args:
            image (Image.Image): 图像对象
            prefix (str, optional): 文件名前缀. Defaults to "temp".
            dir_path (str, optional): 保存目录. Defaults to "./outputs".

        Returns:
            str: 保存文件的绝对路径
        """
        import os
        import time

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.png"
        full_path = os.path.join(dir_path, filename)

        image.save(full_path)
        return full_path