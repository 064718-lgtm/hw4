# SD-Turbo Streamlit Demo

Simple Streamlit app for text-to-image generation inspired by the diffusers notebook in [yenlung/AI-Demo](https://github.com/yenlung/AI-Demo/blob/master/%E3%80%90Demo08%E3%80%91%E7%94%A8diffusers%E5%A5%97%E4%BB%B6%E7%94%9F%E6%88%90%E5%9C%96%E5%83%8F.ipynb). It uses the open `stabilityai/sd-turbo` model, which does not require a Hugging Face token and is optimized for low step counts so it can run on CPU-only Streamlit Cloud.

## Quickstart (local)

1) Use Python 3.10-3.12 (matches Streamlit Cloud runtimes) and create a virtual environment (optional but recommended).
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Run the app:
```
streamlit run app.py
```
4) Open the URL shown in the terminal (defaults to http://localhost:8501).

## Deploy on Streamlit Cloud

1) Push this folder to a GitHub repository.
2) In Streamlit Community Cloud, create a new app and point it to `app.py`.
3) Set the Python version to 3.10+ (the defaults are fine) and keep `requirements.txt` as-is.
4) Optional: set the `Models` directory cache to persist between runs if you want faster cold starts.

## App behavior

- Model: `stabilityai/sd-turbo` via `diffusers.AutoPipelineForText2Image`
- Default settings: 512x512 resolution, 1-8 steps (4 recommended), guidance scale 0-1 recommended
- Supports optional seed for reproducibility
- Attention slicing is enabled to reduce memory on CPU; turbo is trained for very low step counts, so it stays reasonably fast on Streamlit Cloud
- Torch is pinned to 2.5.1 to stay compatible with Streamlit Cloud’s current Python (3.10–3.12).

## Troubleshooting

- If generation is slow, reduce steps to 1-2 and keep 512x512 resolution.
- Out-of-memory on constrained machines: restart the app to clear the cache, or lower resolution (edit `width`/`height` in `app.py` if needed).
- If your region has trouble downloading the model the first time, retry after a few minutes—the model is cached after the first successful load.
