import io
import random
import altair as alt
import pandas as pd
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


def _clean_token(tok: str) -> str:
    """Normalize tokens to avoid artifacts like '</w>' or control tokens."""
    remove_exact = {"<|startoftext|>", "<|endoftext|>", "<|pad|>", "<s>", "</s>", "[PAD]"}
    if tok in remove_exact:
        return ""
    tok = tok.replace("</w>", "")
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("▁", " ")
    tok = tok.strip()
    if tok == "":
        return ""
    return tok


def analyze_prompt_tokens(prompt: str):
    """Approximate per-token importance via encoder embedding norms (avg across encoders when present)."""
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
            tok_clean = _clean_token(tok_str)
            if not tok_clean:
                continue
            token_info.append(
                {
                    "token": tok_clean,
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

    aggregated = {}
    for item in token_info:
        aggregated.setdefault(item["token"], []).append(item["norm"])
    rows = []
    for tok, norms in aggregated.items():
        rows.append({"token": tok, "avg_norm": round(sum(norms) / len(norms), 3)})
    rows.sort(key=lambda x: x["avg_norm"], reverse=True)
    return rows


st.set_page_config(page_title="Diffusers 文生圖 (sd-turbo)", page_icon="🎨", layout="wide", initial_sidebar_state="expanded")
st.title("Diffusers 文生圖 (sd-turbo)")
st.caption("輕量化 Stable Diffusion Turbo：中文介面、範例、Prompt 重要性與下載。")

if "gallery" not in st.session_state:
    st.session_state.gallery = []
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = "A cozy reading nook beside a window with soft morning light, watercolor style"
if "negative_text" not in st.session_state:
    st.session_state.negative_text = "低畫質, 模糊, noisy"
if "steps_val" not in st.session_state:
    st.session_state.steps_val = 4
if "guidance_val" not in st.session_state:
    st.session_state.guidance_val = 0.5

preset_examples = {
    "閱讀角落 (example.png)": {
        "prompt": "A cozy reading nook beside a window with soft morning light, watercolor style",
        "negative": "低畫質, 模糊, noisy",
        "steps": 4,
        "guidance": 0.5,
    },
    "彩虹煙火 (example2.png)": {
        "prompt": "firework with rainbow",
        "negative": "低畫質，模糊，blur",
        "steps": 4,
        "guidance": 0.3,
    },
}

with st.sidebar:
    st.header("操作指南")
    st.markdown(
        """
- 填寫提示詞 / 反向提示，可用中文或英文。
- 建議步數 1-4、引導 0-1；解析度固定 512x512。
- 可輸入種子以重現結果；留空則隨機。
- 首次啟動會下載模型，請稍候。
"""
    )
    preset_choice = st.selectbox("快速載入範例", ["(不套用預設)"] + list(preset_examples.keys()))
    if preset_choice != "(不套用預設)":
        preset = preset_examples[preset_choice]
        st.session_state.prompt_text = preset["prompt"]
        st.session_state.negative_text = preset["negative"]
        st.session_state.steps_val = preset["steps"]
        st.session_state.guidance_val = preset["guidance"]
        st.success(f"已套用預設：{preset_choice}")

tabs = st.tabs(["🖼️ 生成與結果", "📊 Token 重要性", "📄 範例說明", "🗂️ 歷史紀錄"])

with tabs[0]:
    with st.form("generator"):
        st.markdown("**提示設定**")
        prompt = st.text_area(
            "主要提示詞（Prompt）",
            value=st.session_state.prompt_text,
            height=120,
            help="描述你想要的畫面，可以使用中文或英文。",
        )
        negative_prompt = st.text_input(
            "反向提示（避免出現）",
            value=st.session_state.negative_text,
            help="列出不希望出現的元素，逗號分隔。",
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            steps = st.slider(
                "生成步數（Inference steps）",
                min_value=1,
                max_value=8,
                value=st.session_state.steps_val,
                help="sd-turbo 適合 1-4 步，步數越高不一定更好。",
            )
        with col2:
            guidance = st.slider(
                "引導強度（Guidance scale）",
                min_value=0.0,
                max_value=5.0,
                value=st.session_state.guidance_val,
                step=0.1,
                help="建議 0-1；數值越大越貼合提示，但可能出現瑕疵。",
            )
        with col3:
            seed_text = st.text_input("隨機種子（可留空）", placeholder="留空則隨機", help="輸入整數以利重現，留空為隨機。")
            if st.form_submit_button("隨機產生種子", use_container_width=True):
                seed_text = str(random.randint(0, 2**31 - 1))
                st.write(f"本次隨機種子：{seed_text}")

        generate_clicked = st.form_submit_button("生成圖片", use_container_width=True)

    token_rows = []
    generated_image = None
    seed_value = None

    if generate_clicked:
        st.session_state.prompt_text = prompt
        st.session_state.negative_text = negative_prompt
        st.session_state.steps_val = steps
        st.session_state.guidance_val = guidance

        if not prompt.strip():
            st.warning("請輸入提示詞（Prompt）。")
        else:
            if seed_text.strip():
                try:
                    seed_value = int(seed_text.strip())
                except ValueError:
                    st.error("隨機種子需為整數。")
            if seed_text.strip() == "" or seed_value is not None:
                with st.spinner("生成中，請稍候..."):
                    try:
                        generated_image = generate_image(prompt.strip(), negative_prompt.strip(), steps, guidance, seed_value)
                        st.image(generated_image, caption=f"生成結果（sd-turbo） - 種子 {seed_value if seed_value is not None else '隨機'}", use_column_width=True)

                        buffer = io.BytesIO()
                        generated_image.save(buffer, format="PNG")
                        st.download_button(
                            label="下載 PNG",
                            data=buffer.getvalue(),
                            file_name="generated.png",
                            mime="image/png",
                            use_container_width=True,
                        )

                        st.success("已完成生成，可切換到『Token 重要性』或『歷史紀錄』查看。")
                        token_rows = analyze_prompt_tokens(prompt.strip())

                        st.session_state.gallery = (
                            [{"prompt": prompt.strip(), "negative": negative_prompt.strip(), "image_bytes": buffer.getvalue()}]
                            + st.session_state.gallery
                        )[:6]
                    except Exception as e:
                        st.error(f"生成失敗：{e}")

with tabs[1]:
    st.markdown("**Prompt Token 重要性（僅供參考）**")
    st.caption("以 text encoder 輸出向量的 L2 範數近似相對重要性，已清理特殊字元避免亂碼。")

    if not token_rows and st.session_state.get("gallery"):
        st.info("請先在『生成與結果』頁籤完成一次生成以取得 Token 重要性。")
    elif token_rows:
        st.dataframe(token_rows, use_container_width=True)
        df = pd.DataFrame(token_rows)
        chart = (
            alt.Chart(df.head(20))
            .mark_bar()
            .encode(
                x=alt.X("avg_norm:Q", title="平均向量範數 (相對重要性)"),
                y=alt.Y("token:N", sort="-x", title="Token"),
                tooltip=["token", "avg_norm"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("目前尚無可用的 Token 重要性資料。")

with tabs[2]:
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

with tabs[3]:
    st.markdown("**歷史紀錄（最近 6 張）**")
    if not st.session_state.gallery:
        st.info("尚無歷史紀錄，請先生成圖片。")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(st.session_state.gallery):
            col = cols[idx % 3]
            with col:
                st.image(item["image_bytes"], caption=item["prompt"], use_column_width=True)
                st.caption(f"反向提示：{item['negative'] or '(未填)'}")

    if st.button("清除歷史紀錄", type="secondary"):
        st.session_state.gallery = []
        st.success("已清除歷史紀錄。")
