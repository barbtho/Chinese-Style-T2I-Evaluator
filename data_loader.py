# -*- coding: utf-8 -*-
# data_loader.py
import os
import re
from typing import List, Optional

class DataLoader:
    """
    数据加载与预处理模块
    负责从文件系统读取评测数据，并对提示词进行清洗和格式化。
    """

    def __init__(self):
        self.supported_formats = ['.txt', '.csv']

    def load_prompts_from_file(self, file_content: str) -> List[str]:
        """
        从上传的文件内容中解析提示词列表。

        Args:
            file_content (str): 文件解码后的字符串内容

        Returns:
            List[str]: 清洗后的提示词列表
        """
        if not file_content:
            return []

        prompts = []
        raw_lines = file_content.splitlines()

        for line in raw_lines:
            cleaned_line = self._clean_text(line)
            if cleaned_line:
                prompts.append(cleaned_line)

        print(f"数据加载完成: 共解析出 {len(prompts)} 条有效提示词")
        return prompts

    def _clean_text(self, text: str) -> Optional[str]:
        """
        [内部方法] 文本清洗逻辑。
        去除特殊字符、不可见字符及首尾空格。

        Args:
            text (str): 原始文本行

        Returns:
            Optional[str]: 清洗后的文本，如果为空则返回 None
        """
        if not text:
            return None

        # 1. 去除首尾空白
        text = text.strip()

        # 2. 如果是空行或注释行(#开头)，跳过
        if not text or text.startswith('#') or text.startswith('//'):
            return None

        # 3. 去除不可见字符 (如 \u200b 等)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        return text

    def validate_prompt_length(self, prompt: str, min_len: int = 2, max_len: int = 1000) -> bool:
        """
        校验提示词长度是否符合要求。

        Args:
            prompt (str): 提示词
            min_len (int): 最小长度
            max_len (int): 最大长度

        Returns:
            bool: 是否合法
        """
        if not prompt:
            return False
        length = len(prompt)
        return min_len <= length <= max_len

# 示例：预设的测试用例
SAMPLE_PROMPTS = [
    "一幅中国山水画，云雾缭绕",
    "赛博朋克风格的未来城市，霓虹灯",
    "一只在太空漂浮的猫，宇航服"
]