"""
Gradio app inspired by the AI-Demo notebook, now using OpenCV LBPH instead of
DeepFace/TensorFlow for compatibility with Python 3.13 (Streamlit Cloud).
"""
from __future__ import annotations

import os

import gradio as gr

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


def pk(img_path: str, guess_display: str, recognizer, id_to_member) -> str:
    """Compare the AI prediction with the user's guess."""
    predicted_key = predict_member(img_path, recognizer, id_to_member)
    predicted_display = MEMBER_NAME.get(predicted_key, "")

    if not predicted_display:
        return "\u627e\u4e0d\u5230\u5339\u914d\u7d50\u679c\uff0c\u8acb\u78ba\u4fdd photos/ \u5167\u6709\u5404\u6210\u54e1\u7684\u91cf\u8db3\u7167\u7247\uff0c\u4e14\u7167\u7247\u4e2d\u6709\u6b63\u9762\u4eba\u81c9\u3002"

    if guess_display:
        if DISPLAY_TO_KEY.get(guess_display) == predicted_key:
            return f"AI \u4e5f\u8a8d\u70ba\u662f {predicted_display}\uff0c\u4f60\u7b54\u5c0d\u4e86\uff01"
        return f"AI \u8a8d\u70ba\u662f {predicted_display}\uff0c\u4f60\u7684\u7b54\u6848\u662f {guess_display}\u3002"

    return f"AI \u8a8d\u70ba\u662f {predicted_display}\u3002"


def build_interface(recognizer, id_to_member) -> gr.Interface:
    instructions = (
        "\u5148\u5728 photos/<\u6210\u54e1\u82f1\u6587\u540d>/ \u653e\u5165\u8a18\u9304\u7167\u7247\uff08\u4e0d\u540c\u89d2\u5ea6\u3001\u5149\u7dda\uff09\u3002\n"
        "PHOTO_FOLDER \u74b0\u5883\u8b8a\u6578\u53ef\u4ee5\u6539\u8b8a\u8cc7\u6599\u593e\u4f4d\u7f6e\u3002"
    )

    return gr.Interface(
        fn=lambda img_path, guess: pk(img_path, guess, recognizer, id_to_member),
        inputs=[
            gr.Image(label="\u4e0a\u50b3\u7167\u7247 / Upload", type="filepath"),
            gr.Dropdown(MEMBERS_ZH, label="\u4f60\u7684\u731c\u6e2c\uff08\u9078\u586b\uff09"),
        ],
        outputs=gr.Text(label="\u7d50\u679c"),
        title=TITLE,
        description=f"{DESCRIPTION}\n\n{instructions}",
    )


def main() -> None:
    ensure_dataset_dirs()
    recognizer, id_to_member = train_recognizer()
    share = os.environ.get("GRADIO_SHARE", "0") not in {"0", "false", "False"}
    iface = build_interface(recognizer, id_to_member)
    iface.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
