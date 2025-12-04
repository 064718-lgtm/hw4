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
- DeepFace downloads public model weights automatically on first run; no tokens are required.
- `opencv-python-headless` is used to avoid GUI dependencies on headless/Streamlit Cloud environments.
- `tensorflow-cpu==2.15.0` and `keras==2.15.0` are pinned to avoid the Keras 3 incompatibility error that RetinaFace raises on Streamlit Cloud.
- `numpy==1.24.3` / `h5py==3.9.0` are pinned to align with TensorFlow 2.15 wheels.
- Streamlit Cloud: set Python to 3.10.14 (a `.python-version` file is included) so TensorFlow 2.15 installs correctly; Python 3.13 will fail to build TensorFlow.

## Notes
- The interface title/description follows the original Chinese wording via Unicode escapes inside `app.py` to keep the file ASCII-only.
- DeepFace will search the local photo database; if no face is found, it returns a friendly message instead of failing.
- The reference notebook is saved as `demo.ipynb` for traceability.
