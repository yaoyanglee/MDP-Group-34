"""Two-stage pipeline package.
Provides a lightweight OpenCV detector and a tiny classifier.
"""

from .detector import detect_and_crop, detect_and_crop_from_path
from .classifier import TinyClassifier

__all__ = ["detect_and_crop", "detect_and_crop_from_path", "TinyClassifier"]
