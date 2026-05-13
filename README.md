# 🏮 国风文生图评测系统 (Chinese-Style T2I Evaluator)

这是一个专为“中国风”文本生成图像（Text-to-Image）打造的专业、架构清晰的主客观双重评测系统。系统针对 8GB 消费级显卡进行了 Time-Sharing（分时复用）极致优化。

## ✨ 核心亮点
- **双重评测机制**：结合 CLIP 客观物理距离打分与 Qwen 3 主观语义逻辑打分。
- **国风特化**：采用清华系 MiniCPM-V 视觉大模型，精准识别中国风元素。
- **极致显存优化**：通过模型调度策略，在 8G 显存设备上流畅跑通 4 个 AI 大模型。

## 📂 核心文件结构
- `main_advanced.py`: 前端入口 (Streamlit UI)
- `evaluate_advanced.py`: 后端大脑，负责模型调度与评测流水线
- `config.py` & `logger_setup.py`: 全局配置与日志管理

## 🚀 快速开始
```bash
pip install -r requirements.txt
streamlit run main_advanced.py