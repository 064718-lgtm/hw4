# Streamlit Diffusers Text-to-Image (sd-turbo)

Minimal Streamlit app that generates images with Hugging Face `stabilityai/sd-turbo` (a lightweight Stable Diffusion Turbo model) using the diffusers library. No API tokens are required; everything runs locally/CPU on Streamlit Cloud.

## Quickstart (local)
1. `python -m venv .venv && .\.venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Deploy to Streamlit Cloud
1. Push this folder to GitHub.
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud) and select your repo.
3. Set **Main file path** to `app.py` and deploy (the default command `streamlit run app.py` is fine).
4. First run will download model weights; subsequent runs are faster due to caching.

> Note: A `runtime.txt` is included to pin Python 3.11 (currently `python-3.11.6`) on Streamlit Cloud so that prebuilt wheels are available for `tokenizers` and other dependencies.

## Model notes
- Model: `stabilityai/sd-turbo` via `AutoPipelineForText2Image`.
- Optimized for very low inference steps (1–4) and works well with guidance scale near 0–1.
- Resolution fixed at 512x512 to stay within Streamlit Cloud CPU/RAM limits.
- Torch is pinned to `2.5.1`, the latest available on Streamlit Cloud at time of writing, to avoid install errors.

## Files
- `app.py` — Streamlit UI and generation logic.
- `requirements.txt` — Python dependencies for Streamlit Cloud and local use.
- `GPT_Chat.md` — Log of assistant/user conversation per request.
