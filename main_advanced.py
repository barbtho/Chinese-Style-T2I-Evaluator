# -*- coding: utf-8 -*-
# main_advanced.py
import streamlit as st
import time
from evaluate_advanced import TextToImageEvaluator

# 导入可选的辅助模块
try:
    from config import cfg
    from logger_setup import sys_logger
except ImportError:
    pass


def init_session_state() -> None:
    """初始化 Streamlit 的 session_state。"""
    if 'evaluator' not in st.session_state:
        try:
            with st.spinner("系统正在初始化大模型引擎，请稍候..."):
                st.session_state.evaluator = TextToImageEvaluator()
            st.toast("模型加载完成！")
        except Exception as e:
            st.error(f"模型初始化失败: {e}")


def render_sidebar() -> tuple:
    """渲染侧边栏组件。"""
    st.sidebar.header("评测配置")
    input_method = st.sidebar.radio("选择输入方式:", ["单条文本输入", "批量文件输入"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("生成参数")

    width = st.sidebar.selectbox("图像宽度 (Width)", options=[512, 768, 1024], index=0)
    height = st.sidebar.selectbox("图像高度 (Height)", options=[512, 768, 1024], index=0)
    steps = st.sidebar.slider("推理步数 (Steps)", 10, 50, 20)
    scale = st.sidebar.slider("引导系数 (Guidance Scale)", 1.0, 20.0, 7.5)

    return input_method, width, height, steps, scale


def render_input_area(input_method: str) -> list:
    """渲染对应的文本输入框或文件上传器。"""
    prompts = []
    if input_method == "单条文本输入":
        prompt_input = st.text_area("请输入描述词:", placeholder="例如：一幅精美的水墨山水画，远山近松，孤舟泛红", height=150)
        if prompt_input: prompts = [prompt_input]
    else:
        uploaded_file = st.file_uploader("上传评测任务文件 (TXT)", type=['txt'])
        if uploaded_file:
            stringio = uploaded_file.getvalue().decode("utf-8")
            prompts = [line.strip() for line in stringio.splitlines() if line.strip()]
            if prompts: st.info(f"已加载 {len(prompts)} 条评测任务")
    return prompts


def render_results():
    """渲染评测结果展示区域。"""
    if 'results' in st.session_state:
        st.header("深度评测报告")

        for i, result in enumerate(st.session_state.results):
            with st.container():
                st.markdown(f"### 样本 {i + 1}")
                col1, col2 = st.columns([1, 1.2])

                with col1:
                    st.image(
                        result['generated_image'],
                        caption="AI 模型输出",
                        use_container_width=True
                    )

                with col2:
                    st.markdown("#### 文本一致性分析")
                    st.info(f"**原始提示词:** {result['original_text']}")
                    st.success(f"**模型反向描述 (MiniCPM):**\n\n{result['generated_text']}")

                    c_val = result.get('clip_score', 0.0)
                    st.success(f"**CLIP 客观一致性分数:** `{c_val:.4f}`")

                    st.divider()

                    # 4. AI 裁判总分展示
                    ai_res = result['ai_similarity']
                    # 显示整数总分
                    st.markdown(f"#### AI 裁判综合评分: `{int(ai_res['total_score'])}/100`")

                    # 5. 六维度详细指标展示 (百分制整数)
                    m1, m2, m3 = st.columns(3)

                    # 使用整数格式化 (不再显示 .2f)
                    m1.metric("实体对象", f"{int(ai_res['object_score'])}")
                    m2.metric("风格意境", f"{int(ai_res['style_score'])}")
                    m3.metric("数量关系", f"{int(ai_res['quantity_score'])}")

                    st.write("")
                    r2_c1, r2_c2, r2_c3 = st.columns(3)

                    r2_c1.metric("空间方位", f"{int(ai_res['spatial_score'])}")
                    r2_c2.metric("色彩准确", f"{int(ai_res['color_score'])}")
                    r2_c3.metric("语义一致", f"{int(ai_res['overall_semantic_score'])}")

                    with st.expander("查看 AI 评分详细理由"):
                        st.write(ai_res['reasoning'])

            st.divider()


def main():
    """Streamlit 应用主入口函数。"""
    st.set_page_config(page_title="文生图模型评测系统", layout="wide")
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4; font-family: Microsoft YaHei;'>文生图模型评测系统</h1>",
        unsafe_allow_html=True)

    init_session_state()
    input_method, width, height, steps, scale = render_sidebar()
    prompts = render_input_area(input_method)

    if st.sidebar.button("开始自动化评测", type="primary"):
        if not prompts:
            st.warning("请先输入提示词或上传文件")
            return

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(prompts)
        for i, p in enumerate(prompts):
            status_text.text(f"正在处理第 {i + 1}/{total} 条: {p[:20]}...")
            res = st.session_state.evaluator.evaluate_single_prompt(p, width, height, steps, scale)
            results.append(res)
            progress_bar.progress((i + 1) / total)

        st.session_state.results = results
        status_text.success("评测任务全部完成！")

    render_results()


if __name__ == "__main__":
    main()