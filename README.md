# IVE Face PK (DeepFace + Gradio)

Small project inspired by the notebook `和_AI_PK_看誰比較會認_IVE_成員.ipynb`. It builds a local face-recognition quiz using [DeepFace](https://github.com/serengil/deepface) and [Gradio](https://www.gradio.app/) so you can compare your guess against the model.

## Setup
1. Install Python 3.9+ (DeepFace/TensorFlow are heavy; a virtual env is recommended).
2. Install dependencies:
   ```bash
   python -m venv .venv
   .\\.venv\\Scripts\\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Prepare the photo database
- Reference images live under `photos/<member>/`. The default members are:
  `yujin`, `wonyoung`, `gaeul`, `rei`, `liz`, `leeseo`.
- Put several clear, front-facing photos for each member in their folder. (The app auto-creates empty folders on first run.)
- To use a different root folder, set `PHOTO_FOLDER=/path/to/photos` before launching.
- After adding photos, restart the app (or click reload in Streamlit) so the LBPH model retrains on the new data.

## Run the app (Gradio)
```bash
python app.py
```
- Open the Gradio URL from the terminal. Upload a quiz photo, pick your guess, and see whether the AI agrees.
- Set `GRADIO_SHARE=1` if you need a public share URL.

## Run the app (Streamlit)
```bash
streamlit run streamlit_app.py
```
- Opens a Streamlit UI; upload a photo and optionally select your guess.
- Uses OpenCV LBPH (no TensorFlow/Keras), so it works on Streamlit Cloud with Python 3.13+.
- `opencv-contrib-python-headless` is used to avoid GUI dependencies on headless environments.

## Notes
- The interface title/description follows the original Chinese wording via Unicode escapes inside `app.py` / `streamlit_app.py` to keep the files ASCII-only.
- The backend now uses OpenCV LBPH; restart/reload after adding new photos to rebuild the recognizer.
- The reference notebook is saved as `demo.ipynb` for traceability.
