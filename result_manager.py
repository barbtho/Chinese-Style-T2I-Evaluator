# result_manager.py
import json
import csv
import os
import time
from datetime import datetime


class ResultManager:
    """
    评测结果管理模块
    负责评测数据的持久化存储、导出和格式化
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.ensure_directory()

    def ensure_directory(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except OSError as e:
                print(f"创建目录失败: {e}")

    def generate_filename(self, prefix="eval", ext="json"):
        """生成带时间戳的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{ext}"

    def save_to_json(self, results, filename=None):
        """
        将评测结果保存为JSON格式
        :param results: 评测结果列表
        :param filename: 可选文件名
        """
        if filename is None:
            filename = self.generate_filename(prefix="report", ext="json")

        filepath = os.path.join(self.output_dir, filename)

        # 预处理数据，移除不可序列化的对象（如图片Tensor或PIL对象）
        serializable_results = []
        for res in results:
            item = res.copy()
            # 移除图片对象，避免JSON序列化报错
            if 'generated_image' in item:
                del item['generated_image']
            serializable_results.append(item)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": len(results),
                    "data": serializable_results
                }, f, ensure_ascii=False, indent=4)
            print(f"结果已保存至 JSON: {filepath}")
            return filepath
        except Exception as e:
            print(f"JSON保存失败: {e}")
            return None

    def save_to_csv(self, results, filename=None):
        """
        将评测结果保存为CSV格式（便于Excel查看）
        """
        if not results:
            return None

        if filename is None:
            filename = self.generate_filename(prefix="report", ext="csv")

        filepath = os.path.join(self.output_dir, filename)

        try:
            # 提取扁平化字段
            # 假设 results 结构包含嵌套字典，需要展平
            flat_results = []
            for res in results:
                flat_row = {
                    "original_text": res.get("original_text", ""),
                    "english_text": res.get("english_text", ""),
                    "generated_text": res.get("generated_text", ""),
                    "clip_score": res.get("clip_score", 0),
                }

                # 提取AI评分详情
                ai_data = res.get("ai_similarity", {})
                if ai_data:
                    flat_row["ai_total_score"] = ai_data.get("total_score", 0)
                    flat_row["object_score"] = ai_data.get("object_score", 0)
                    flat_row["style_score"] = ai_data.get("style_score", 0)
                    flat_row["ai_comment"] = ai_data.get("reasoning", "")

                flat_results.append(flat_row)

            if not flat_results:
                return None

            # 获取表头
            headers = flat_results[0].keys()

            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(flat_results)

            print(f"结果已保存至 CSV: {filepath}")
            return filepath

        except Exception as e:
            print(f"CSV保存失败: {e}")
            return None