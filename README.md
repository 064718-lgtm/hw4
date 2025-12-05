# Streamlit Diffusers Text-to-Image (sd-turbo)

Minimal Streamlit app that generates images with Hugging Face `stabilityai/sd-turbo` (a lightweight Stable Diffusion Turbo model) using the diffusers library. No API tokens are required; everything runs locally/CPU on Streamlit Cloud.

## Live Demo
- Streamlit: https://43onov79zc27hey78w7tzk.streamlit.app/

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
- `transformers==4.37.2` with `tokenizers==0.15.2` (prebuilt wheels for Python 3.11) to avoid Rust builds on Streamlit Cloud; `diffusers==0.27.2` remains compatible with sd-turbo.
- `huggingface_hub==0.20.3` is pinned to keep `cached_download` available for diffusers 0.27.x.

## Model notes
- Model: `stabilityai/sd-turbo` via `AutoPipelineForText2Image`.
- Optimized for very low inference steps (1–4) and works well with guidance scale near 0–1.
- Resolution fixed at 512x512 to stay within Streamlit Cloud CPU/RAM limits.
- Torch is pinned to `2.5.1`, the latest available on Streamlit Cloud at time of writing, to avoid install errors.

## UI 說明
- 主頁籤「生成與結果」：中文說明，提示詞/反向提示、步數、引導強度、隨機種子（可隨機產生），並可下載 PNG。
- 「Token 重要性」：生成後顯示表格與前 20 名條狀圖，基於 text encoder 向量範數，已清理特殊字元以避免亂碼。
- 「範例說明」：展示 `example.png`（閱讀角落）與 `example2.png`（彩虹煙火），並列出對應提示詞、反向提示與建議步數/引導。
- 「歷史紀錄」：保留最近 6 張結果，附上提示詞與反向提示，可一鍵清除。
- 側邊欄提供快速套用範例設定與操作提醒。

## Files
- `app.py` — Streamlit UI and generation logic.
- `requirements.txt` — Python dependencies for Streamlit Cloud and local use.
- `runtime.txt` — Pins Python version for Streamlit Cloud.
- `example.png` — 範例圖片，顯示在「範例說明」頁面。
- `GPT_Chat.md` — Log of assistant/user conversation per request.
