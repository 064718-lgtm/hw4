import io
import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

MODEL_ID = "stabilityai/sd-turbo"


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the lightweight text-to-image pipeline once per session."""
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cpu")


def generate_image(prompt: str, negative_prompt: str, steps: int, guidance: float, seed: int | None):
    pipe = load_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        height=512,
        width=512,
    )
    return result.images[0]


st.set_page_config(page_title="Diffusers 文生圖 (sd-turbo)", page_icon="🎨", layout="wide")
st.title("Diffusers 文生圖 (sd-turbo)")
st.caption("輕量化 Stable Diffusion Turbo：提供中文說明、範例圖片與快速生成介面。")

tabs = st.tabs(["🖼️ 生成圖片", "📄 範例說明"])

with tabs[0]:
    st.markdown(
        """
**使用說明（中文）：**
- 輸入想要生成的描述（Prompt），可用中文或英文。
- 可填寫「反向提示」避免出現的元素，如「低畫質、模糊」。
- sd-turbo 建議步數 1-4、Guidance 0-1，解析度固定 512x512 以適應 Streamlit Cloud CPU。
- 首次啟動需下載模型，請稍候。
"""
    )

    with st.form("generator"):
        prompt = st.text_area(
            "主要提示詞（Prompt）",
            value="A cozy reading nook beside a window with soft morning light, watercolor style",
            height=100,
            help="描述你想要的畫面，可以使用中文或英文。",
        )
        negative_prompt = st.text_input(
            "反向提示（避免出現）", placeholder="低畫質, 模糊, noisy", help="列出不希望出現的元素，逗號分隔。"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            steps = st.slider(
                "生成步數（Inference steps）",
                min_value=1,
                max_value=8,
                value=4,
                help="sd-turbo 適合 1-4 步，步數越高不一定更好。",
            )
        with col2:
            guidance = st.slider(
                "引導強度（Guidance scale）",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                help="建議 0-1；數值越大越貼合提示，但可能出現瑕疵。",
            )
        with col3:
            seed_text = st.text_input("隨機種子（可留空）", placeholder="留空則隨機", help="輸入整數以利重現，留空為隨機。")

        generate_clicked = st.form_submit_button("生成圖片", use_container_width=True)

    if generate_clicked:
        if not prompt.strip():
            st.warning("請輸入提示詞（Prompt）。")
        else:
            seed_value = None
            if seed_text.strip():
                try:
                    seed_value = int(seed_text.strip())
                except ValueError:
                    st.error("隨機種子需為整數。")
            if seed_text.strip() == "" or seed_value is not None:
                with st.spinner("生成中，請稍候..."):
                    try:
                        image = generate_image(prompt.strip(), negative_prompt.strip(), steps, guidance, seed_value)
                        st.image(image, caption="生成結果（sd-turbo）", use_column_width=True)

                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        st.download_button(
                            label="下載 PNG",
                            data=buffer.getvalue(),
                            file_name="generated.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"生成失敗：{e}")

with tabs[1]:
    st.markdown(
        """
**範例流程（Example Walkthrough）：**
1. 在「生成圖片」頁籤輸入提示詞：`A cozy reading nook beside a window with soft morning light, watercolor style`。
2. 反向提示：`低畫質, 模糊, noisy` 以減少不想要的雜訊。
3. 建議步數 4、引導 0.5，點擊「生成圖片」。
4. 生成完成後可以直接下載 PNG。
"""
    )
    st.image("example.png", caption="example.png 範例輸出示意", use_column_width=True)
    st.info("首次啟動會下載模型，若等待較久屬正常現象。若需內容過濾，請啟用安全檢查或另行加上審核。")
