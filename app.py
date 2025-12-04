"""
Gradio app inspired by the AI-Demo notebook:
https://github.com/yenlung/AI-Demo/blob/master/%E3%80%90Demo03%E3%80%91%E5%92%8C_AI_PK_%E7%9C%8B%E8%AA%B0%E6%AF%94%E8%BC%83%E6%9C%83%E8%AA%8D_IVE_%E6%88%90%E5%93%A1.ipynb
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from deepface import DeepFace
import gradio as gr

# Folder name keys are kept in ASCII to avoid path issues on different OSes.
MEMBERS_EN: List[str] = ["yujin", "wonyoung", "gaeul", "rei", "liz", "leeseo"]

# Display names are in Unicode but represented with escapes to keep this file ASCII-only.
MEMBERS_ZH: List[str] = [
    "\u516a\u771f\u110b\u1172\u110c\u1175\u11ab",
    "\u54e1\u745b\u110b\u116f\u11ab\u110b\u1167\u11bc",
    "\u79cb\u5929\u1100\u1161\u110b\u1173\u11af",
    "Liz\u1105\u1175\u110c\u1173",
    "Rei\u1105\u1166\u110b\u1175",
    "\u674e\u745e\u110b\u1175\u1109\u1165",
]

TITLE = "IVE \u6210\u54e1 PK \u6230"
DESCRIPTION = "\u548c AI PK IVE \u6210\u54e1\u7684\u8fa8\u8b58\u80fd\u529b"

PHOTO_FOLDER = Path(os.environ.get("PHOTO_FOLDER", "photos")).resolve()

MEMBER_NAME: Dict[str, str] = dict(zip(MEMBERS_EN, MEMBERS_ZH))
DISPLAY_TO_KEY: Dict[str, str] = {v: k for k, v in MEMBER_NAME.items()}


def ensure_dataset_dirs() -> None:
    """Create one folder per member to store reference photos."""
    for member in MEMBERS_EN:
        (PHOTO_FOLDER / member).mkdir(parents=True, exist_ok=True)


def recognize_member(img_path: str) -> str:
    """
    Run DeepFace search against the local photo folder.

    Returns the matched member key or an empty string if no match is found.
    """
    try:
        df_list = DeepFace.find(img_path, db_path=str(PHOTO_FOLDER), enforce_detection=False)
        if not df_list:
            return ""
        df = df_list[0]
        if df.empty:
            return ""
        best_match = Path(df.identity.iloc[0])
        return best_match.parent.name
    except Exception as exc:  # DeepFace raises many broad exceptions internally
        print(f"[warn] recognition failed: {exc}")
        return ""


def pk(img_path: str, guess_display: str) -> str:
    """Compare the AI prediction with the user's guess."""
    predicted_key = recognize_member(img_path)
    predicted_display = MEMBER_NAME.get(predicted_key, "")

    if not predicted_display:
        return "\u627e\u4e0d\u5230\u5339\u914d\u7d50\u679c\uff0c\u8acb\u78ba\u4fdd photos/ \u5167\u6709\u5404\u6210\u54e1\u7684\u91cf\u8db3\u7167\u7247\uff0c\u4e14\u7167\u7247\u4e2d\u6709\u6b63\u9762\u4eba\u81c9\u3002"

    if guess_display:
        if DISPLAY_TO_KEY.get(guess_display) == predicted_key:
            return f"AI \u4e5f\u8a8d\u70ba\u662f {predicted_display}\uff0c\u4f60\u7b54\u5c0d\u4e86\uff01"
        return f"AI \u8a8d\u70ba\u662f {predicted_display}\uff0c\u4f60\u7684\u7b54\u6848\u662f {guess_display}\u3002"

    return f"AI \u8a8d\u70ba\u662f {predicted_display}\u3002"


def build_interface() -> gr.Interface:
    instructions = (
        "\u5148\u5728 photos/<\u6210\u54e1\u82f1\u6587\u540d>/ \u653e\u5165\u8a18\u9304\u7167\u7247\uff08\u4e0d\u540c\u89d2\u5ea6\u3001\u5149\u7dda\uff09\u3002\n"
        "PHOTO_FOLDER \u74b0\u5883\u8b8a\u6578\u53ef\u4ee5\u6539\u8b8a\u8cc7\u6599\u593e\u4f4d\u7f6e\u3002"
    )

    return gr.Interface(
        fn=pk,
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
    share = os.environ.get("GRADIO_SHARE", "0") not in {"0", "false", "False"}
    iface = build_interface()
    iface.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
