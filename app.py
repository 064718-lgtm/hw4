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


def analyze_prompt_tokens(prompt: str):
    """Approximate per-token importance via encoder embedding norms (avg across SDXL dual encoders when present)."""
    pipe = load_pipeline()
    token_info = []

    def collect(tok, enc):
        inputs = tok(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=tok.model_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            outputs = enc(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # [1, seq, dim]
            norms = hidden.norm(dim=-1).squeeze(0).cpu().tolist()

        tokens = tok.convert_ids_to_tokens(input_ids[0])
        mask_list = attention_mask[0].tolist()
        for tok_str, norm, mask in zip(tokens, norms, mask_list):
            if mask == 0:
                continue
            if tok_str in ("<pad>", "</s>", "<|endoftext|>"):
                continue
            token_info.append(
                {
                    "token": tok_str,
                    "norm": round(norm, 3),
                }
            )

    try:
        if hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder"):
            collect(pipe.tokenizer, pipe.text_encoder)
        if hasattr(pipe, "tokenizer_2") and hasattr(pipe, "text_encoder_2"):
            collect(pipe.tokenizer_2, pipe.text_encoder_2)
    except Exception:
        return []

    # Aggregate by token string (average if duplicated due to dual encoders)
    aggregated = {}
    for item in token_info:
        aggregated.setdefault(item["token"], []).append(item["norm"])
    rows = []
    for tok, norms in aggregated.items():
        rows.append({"token": tok, "avg_norm": round(sum(norms) / len(norms), 3)})
    rows.sort(key=lambda x: x["avg_norm"], reverse=True)
    return rows


st.set_page_config(page_title="Diffusers 文生圖 (sd-turbo)", page_icon="🎨", layout="wide")
st.title("Diffusers 文生圖 (sd-turbo)")
st.caption("輕量化 Stable Diffusion Turbo：中文介面、範例圖片與快速生成。")

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

                        st.subheader("Prompt Token 重要性（嵌入向量強度近似）")
                        st.caption("以下為 text encoder 輸出向量的 L2 範數平均值，僅作為相對重要性參考。")
                        token_rows = analyze_prompt_tokens(prompt.strip())
                        if token_rows:
                            st.dataframe(token_rows, use_container_width=True)
                        else:
                            st.info("無法取得 token 重要性（可能是模型或環境不支援）。")
                    except Exception as e:
                        st.error(f"生成失敗：{e}")

with tabs[1]:
    st.markdown(
        """
**範例流程（Example Walkthrough）：**
- 範例 1：
  - 主要提示詞：`A cozy reading nook beside a window with soft morning light, watercolor style`
  - 反向提示：`低畫質, 模糊, noisy`
  - 建議步數：4，建議引導：0.5
- 範例 2：
  - 主要提示詞：`firework with rainbow`
  - 反向提示：`低畫質，模糊，blur`
  - 建議步數：4，建議引導：0.3
"""
    )
    st.image("example.png", caption="example.png 範例 1 輸出示意", use_column_width=True)
    st.image("example2.png", caption="example2.png 範例 2 輸出示意", use_column_width=True)
    st.info("首次啟動會下載模型，若等待較久屬正常現象。若需內容過濾，請啟用安全檢查或另行加上審核。")
