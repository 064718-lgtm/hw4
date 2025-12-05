"""
Lightweight face recognition backend using OpenCV LBPH to avoid TensorFlow/Keras
dependency issues on Streamlit Cloud (Python 3.13).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Folder name keys are kept in ASCII to avoid path issues on different OSes.
MEMBERS_EN: List[str] = ["yujin", "wonyoung", "gaeul", "rei", "liz", "leeseo"]

# Display names (Unicode escaped to keep file ASCII).
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

# LBPH returns lower confidence for better matches; keep a conservative threshold.
DEFAULT_CONFIDENCE_THRESHOLD = 80.0
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dataset_dirs(photo_root: Path = PHOTO_FOLDER) -> None:
    """Create one folder per member to store reference photos (no-op if absent)."""
    for member in MEMBERS_EN:
        (photo_root / member).mkdir(parents=True, exist_ok=True)


def _load_image_as_gray(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    return resized


def load_training_data(photo_root: Path = PHOTO_FOLDER) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    id_to_member: Dict[int, str] = {}
    for idx, member in enumerate(MEMBERS_EN):
        id_to_member[idx] = member
        folder = photo_root / member
        if not folder.exists():
            continue
        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            img = _load_image_as_gray(img_path)
            if img is None:
                continue
            images.append(img)
            labels.append(idx)
    return images, labels, id_to_member


def train_recognizer(photo_root: Path = PHOTO_FOLDER) -> Tuple[Optional[cv2.face_LBPHFaceRecognizer], Dict[int, str]]:
    images, labels, id_to_member = load_training_data(photo_root)
    if not images:
        return None, id_to_member
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer, id_to_member


def predict_member(
    img_path: str,
    recognizer: Optional[cv2.face_LBPHFaceRecognizer],
    id_to_member: Dict[int, str],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """
    Predict member key using LBPH recognizer; returns "" if not confident.
    """
    if recognizer is None:
        return ""
    img = _load_image_as_gray(Path(img_path))
    if img is None:
        return ""
    label, confidence = recognizer.predict(img)
    if confidence > confidence_threshold:
        return ""
    return id_to_member.get(label, "")
