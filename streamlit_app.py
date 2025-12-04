"""
Streamlit version of the IVE face PK app, now using OpenCV LBPH instead of
TensorFlow/DeepFace to avoid Python 3.13 wheel issues on Streamlit Cloud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from face_backend import (
    DESCRIPTION,
    DISPLAY_TO_KEY,
    MEMBERS_ZH,
    MEMBER_NAME,
    TITLE,
    ensure_dataset_dirs,
    predict_member,
    train_recognizer,
)


@st.cache_resource
def load_model():
    return train_recognizer()


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
    ensure_dataset_dirs()

    st.set_page_config(page_title=TITLE, page_icon=":camera:")
    st.title(TITLE)
    st.write(DESCRIPTION)
    st.info(
        "\u5148\u5728 photos/<\u6210\u54e1\u82f1\u6587\u540d>/ \u653e\u5165\u8a18\u9304\u7167\u7247\uff08\u4e0d\u540c\u89d2\u5ea6\u3001\u5149\u7dda\uff09\u3002\n"
        "PHOTO_FOLDER \u74b0\u5883\u8b8a\u6578\u53ef\u4ee5\u6539\u8b8a\u8cc7\u6599\u593e\u4f4d\u7f6e\u3002\n"
        "OpenCV LBPH \u4e0d\u4f9d\u8cf4 TensorFlow/\u6d3e\u751f\u78c1\u78bc\u6a5f\u5668\uff0c\u9069\u7528\u65bc Streamlit Cloud Python 3.13\u3002"
    )

    recognizer, id_to_member = load_model()

    if st.button("\u91cd\u65b0\u8f09\u5165\u8cc7\u6599\u96c6 / Reload dataset"):
        load_model.clear()
        recognizer, id_to_member = load_model()
        st.toast("\u5df2\u91cd\u8f09\u8cc7\u6599\u96c6", icon="✅")

    uploaded = st.file_uploader("\u4e0a\u50b3\u4f60\u8981 PK \u7684\u7167\u7247 (jpg/png)", type=["jpg", "jpeg", "png"])
    guess = st.selectbox("\u9078\u64c7\u4f60\u7684\u731c\u6e2c (可空)", [""] + MEMBERS_ZH)

    if uploaded:
        img_path = Path("temp_upload.jpg")
        img_path.write_bytes(uploaded.getbuffer())

        with st.spinner("\u8655\u7406\u4e2d..."):
            predicted_key = predict_member(str(img_path), recognizer, id_to_member)
        st.success(describe_result(predicted_key, guess or None))

        st.image(str(img_path), caption="Uploaded")
    else:
        st.warning("\u8acb\u5148\u4e0a\u50b3\u7167\u7247\u3002")


if __name__ == "__main__":
    main()
