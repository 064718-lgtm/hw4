from io import BytesIO

import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler


st.set_page_config(page_title="SD-Turbo Image Generator", page_icon="🎨")
torch.set_grad_enabled(False)


@st.cache_resource(show_spinner="Loading the lightweight model...")
def load_pipeline(model_id: str = "stabilityai/sd-turbo"):
    """
    Load and cache the SD-Turbo text-to-image pipeline.
    Uses float16 on GPU for speed and float32 on CPU for compatibility.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe, device


pipe, device = load_pipeline()

st.title("SD-Turbo on Streamlit Cloud")
st.caption(
    "Lightweight image generation using diffusers with the open `stabilityai/sd-turbo` model. "
    "Optimized for low step counts so it can run on CPU-only Streamlit Cloud."
)

with st.expander("How to get good results", expanded=False):
    st.markdown(
        "- Keep prompts concise but specific (style + subject + scene + mood).\n"
        "- SD-Turbo prefers 1-4 inference steps; try 4 for better detail.\n"
        "- Guidance scale works best around 0-1; higher values may oversaturate.\n"
        "- Use 512x512 resolution for the best speed/quality trade-off."
    )


prompt = st.text_area(
    "Prompt",
    value="A cozy watercolor illustration of a small cabin in the mountains at sunrise, soft pastel colors",
    height=100,
)
negative_prompt = st.text_area(
    "Negative prompt (optional)",
    value="blurry, distorted, low quality, text, watermark",
    height=80,
)

col1, col2, col3 = st.columns(3)
num_inference_steps = col1.slider("Inference steps", min_value=1, max_value=8, value=4, help="SD-Turbo was trained for 1-4 steps.")
guidance_scale = col2.slider("Guidance scale", min_value=0.0, max_value=3.0, value=0.0, step=0.1, help="0-1 recommended for SD-Turbo.")
use_seed = col3.checkbox("Set seed", value=False)
seed_value = col3.number_input("Seed value", min_value=0, max_value=2_147_483_647, value=0, step=1, disabled=not use_seed)


def make_generator(selected_device: str, enabled: bool, seed: int | None):
    if not enabled:
        return None
    return torch.Generator(device=selected_device).manual_seed(int(seed))


generate = st.button("Generate", type="primary", use_container_width=True, disabled=not prompt.strip())

if generate:
    with st.spinner("Generating... this may take ~10-40s on CPU"):
        try:
            generator = make_generator(device, use_seed, seed_value if use_seed else None)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=512,
                height=512,
                generator=generator,
            )
            image = result.images[0]
            st.image(image, caption="Generated image", use_column_width=True)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            st.download_button(
                "Download PNG",
                data=buffer.getvalue(),
                file_name="sd_turbo_image.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception as exc:  # pragma: no cover - UI surface
            st.error(f"Generation failed: {exc}")
