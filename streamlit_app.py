"""
Streamlit version of the IVE face PK app, now using OpenCV LBPH instead of
TensorFlow/DeepFace to avoid Python 3.13 wheel issues on Streamlit Cloud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil
import tempfile
import zipfile

import streamlit as st

from face_backend import (
    DESCRIPTION,
    DISPLAY_TO_KEY,
    MEMBERS_ZH,
    MEMBER_NAME,
    PHOTO_FOLDER,
    TITLE,
    ensure_dataset_dirs,
    predict_member,
    train_recognizer,
)


@st.cache_resource
def load_model(photo_root_str: str):
    return train_recognizer(Path(photo_root_str))


def _extract_reference_zip(uploaded_file) -> Optional[Path]:
    """Extract uploaded ZIP to a temp folder and return its path."""
    if uploaded_file is None:
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="photos_"))
    zip_path = tmp_dir / "refs.zip"
    zip_path.write_bytes(uploaded_file.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    # Expect subfolders matching member names; user is responsible for structure.
    return tmp_dir


def describe_result(predicted_key: str, user_guess: Optional[str]) -> str:
    predicted_display = MEMBER_NAME.get(predicted_key, "")
    if not predicted_display:
        return "\u627e\u4e0d\u5230\u5339\u914d\u7d50\u679c\uff0c\u8acb\u78ba\u4fdd photos/ \u5167\u6709\u5404\u6210\u54e1\u7684\u91cf\u8db3\u7167\u7247\uff0c\u4e14\u7167\u7247\u4e2d\u6709\u6b63\u9762\u4eba\u81c9\u3002"

    if user_guess:
        if DISPLAY_TO_KEY.get(user_guess) == predicted_key:
            return f"AI \u4e5f\u8a8d\u70ba\u662f {predicted_display}\uff0c\u4f60\u7b54\u5c0d\u4e86\uff01"
        return f"AI \u8a8d\u70ba\u662f {predicted_display}\uff0c\u4f60\u7684\u7b54\u6848\u662f {user_guess}\u3002"

    return f"AI \u8a8d\u70ba\u662f {predicted_display}\u3002"


def main() -> None:
    # Track current photo root in session; default to PHOTO_FOLDER (env or ./photos)
    if "photo_root" not in st.session_state:
        st.session_state.photo_root = PHOTO_FOLDER
    photo_root: Path = st.session_state.photo_root

    ensure_dataset_dirs(photo_root)

    st.set_page_config(page_title=TITLE, page_icon=":camera:")
    st.title(TITLE)
    st.write(DESCRIPTION)
    st.info(
        "\u5148\u5728 photos/<\u6210\u54e1\u82f1\u6587\u540d>/ \u653e\u5165\u8a18\u9304\u7167\u7247\uff08\u4e0d\u540c\u89d2\u5ea6\u3001\u5149\u7dda\uff09\u3002\n"
        "PHOTO_FOLDER \u74b0\u5883\u8b8a\u6578\u53ef\u4ee5\u6539\u8b8a\u8cc7\u6599\u593e\u4f4d\u7f6e\u3002\n"
        "\u4e5f\u53ef\u4ee5\u4e0a\u50b3 ZIP\uff08\u5167\u542b photos/<\u6210\u54e1\u82f1\u6587\u540d>/\u7167\u7247\uff09\u5373\u5831\u73fe\u5730\u8f49\u63db\uff0c\u7121\u9700\u5c0e\u5165\u6574\u500b\u8cc7\u6599\u96c6\u5230 repo\u3002\n"
        "OpenCV LBPH \u4e0d\u4f9d\u8cf4 TensorFlow/\u6d3e\u751f\u78c1\u78bc\u6a5f\u5668\uff0c\u9069\u7528\u65bc Streamlit Cloud Python 3.13\u3002"
    )

    st.subheader("\u53c3\u8003\u7167\u7247\u7ba1\u7406")
    ref_zip = st.file_uploader("\u4e0a\u50b3\u5305\u542b photos/<\u6210\u54e1\u82f1\u6587\u540d>/ \u7d44\u7e54\u7684 ZIP", type=["zip"])
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\u532f\u5165 ZIP \u4e26\u91cd\u65b0\u8a13\u7df4"):
            if ref_zip is None:
                st.warning("\u8acb\u4e0a\u50b3 ZIP\u3002")
            else:
                if "tmp_photo_root" in st.session_state:
                    shutil.rmtree(st.session_state.tmp_photo_root, ignore_errors=True)
                tmp_root = _extract_reference_zip(ref_zip)
                st.session_state.photo_root = tmp_root
                st.session_state.tmp_photo_root = tmp_root
                load_model.clear()
                st.success("\u5df2\u532f\u5165 ZIP \u4e26\u91cd\u5efa\u6a21\u578b")
    with col2:
        if st.button("\u4f7f\u7528\u9810\u8a2d PHOTO_FOLDER \u91cd\u65b0\u8a13\u7df4"):
            st.session_state.photo_root = PHOTO_FOLDER
            load_model.clear()
            st.success("\u5df2\u5207\u63db\u70ba\u9810\u8a2d\u8cc7\u6599\u593e")

    photo_root = Path(st.session_state.photo_root)
    recognizer, id_to_member = load_model(str(photo_root))

    if st.button("\u91cd\u65b0\u8f09\u5165\u8cc7\u6599\u96c6 / Reload dataset"):
        load_model.clear()
        recognizer, id_to_member = load_model()
        st.toast("\u5df2\u91cd\u8f09\u8cc7\u6599\u96c6", icon="✅")

    uploaded = st.file_uploader("\u4e0a\u50b3\u4f60\u8981 PK \u7684\u7167\u7247 (jpg/png)", type=["jpg", "jpeg", "png"])
    guess = st.selectbox("\u9078\u64c7\u4f60\u7684\u731c\u6e2c (可空)", [""] + MEMBERS_ZH)

    if uploaded:
        img_path = Path("temp_upload.jpg")
        img_path.write_bytes(uploaded.getbuffer())

        if recognizer is None:
            st.error("\u5c1a\u672a\u627e\u5230\u4efb\u4f55\u53c3\u8003\u7167\u7247\uff0c\u8acb\u5148\u532f\u5165 ZIP \u6216\u5efa\u7acb photos/<\u6210\u54e1\u540d>/ \u653e\u4e0a\u7167\u7247\u3002")
        else:
            with st.spinner("\u8655\u7406\u4e2d..."):
                predicted_key = predict_member(str(img_path), recognizer, id_to_member)
            st.success(describe_result(predicted_key, guess or None))

        st.image(str(img_path), caption="Uploaded")
    else:
        st.warning("\u8acb\u5148\u4e0a\u50b3\u7167\u7247\u3002")


if __name__ == "__main__":
    main()
