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


st.set_page_config(page_title="Diffusers Text-to-Image", page_icon="🎨", layout="wide")
st.title("Text-to-Image on Streamlit (sd-turbo)")
st.caption("Quick demo inspired by yenlung/AI-Demo diffusers notebook, using a lightweight sd-turbo pipeline.")

with st.form("generator"):
    prompt = st.text_area(
        "Prompt",
        value="A cozy reading nook beside a window with soft morning light, watercolor style",
        height=100,
    )
    negative_prompt = st.text_input("Negative prompt (optional)", placeholder="low quality, blurry, noisy")

    col1, col2, col3 = st.columns(3)
    with col1:
        steps = st.slider("Inference steps", min_value=1, max_value=8, value=4, help="sd-turbo works best with 1-4 steps.")
    with col2:
        guidance = st.slider("Guidance scale", min_value=0.0, max_value=5.0, value=0.0, step=0.1, help="Try 0-1 for sd-turbo.")
    with col3:
        seed_text = st.text_input("Seed (optional, int)", placeholder="Leave blank for random")

    generate_clicked = st.form_submit_button("Generate", use_container_width=True)

if generate_clicked:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        seed_value = None
        if seed_text.strip():
            try:
                seed_value = int(seed_text.strip())
            except ValueError:
                st.error("Seed must be an integer.")
        if seed_text.strip() == "" or seed_value is not None:
            with st.spinner("Generating image..."):
                try:
                    image = generate_image(prompt.strip(), negative_prompt.strip(), steps, guidance, seed_value)
                    st.image(image, caption="Generated with sd-turbo", use_column_width=True)

                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    st.download_button(
                        label="Download PNG",
                        data=buffer.getvalue(),
                        file_name="generated.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")

with st.expander("Usage tips"):
    st.markdown(
        """
- sd-turbo is optimized for very few steps; start with 4 steps and guidance near 0-1.
- Prompts with clear subjects and styles converge faster (e.g., "isometric pixel art of a neon city alley").
- The first run downloads model weights; expect a short wait on cold start.
- Streamlit Cloud runs on CPU, so keep resolution at 512x512 for speed and memory.
"""
    )
