import cv2
from qreader import QReader
from PIL import Image
import torch

def check_scannable(image_path: str) -> str | None:
    """Check if the generated QR code is scannable. Returns decoded text or None."""
    try:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        decoded = qreader.detect_and_decode(image=img)
        return decoded[0] if decoded else None
    except Exception:
        return None