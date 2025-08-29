"""OpenCV-based lightweight detector for square stickers.

Functions:
 - detect_and_crop(frame): returns list of cropped images (BGR numpy arrays)
 - detect_and_crop_from_path(path): convenience for single image path

Heuristics: grayscale -> blur -> adaptive threshold -> find contours -> filter by quadrilateral
"""
from typing import List, Tuple
import cv2
import numpy as np


def _order_pts(pts: np.ndarray) -> np.ndarray:
    # order points: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_pts(pts)
    (tl, tr, br, bl) = rect
    # compute width and height of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_and_crop(frame: np.ndarray, min_area: int = 500, eps_coef: float = 0.02) -> List[np.ndarray]:
    """Detect quadrilateral stickers in BGR frame and return cropped BGR images.

    Args:
        frame: BGR image (numpy array)
        min_area: minimum contour area to keep
        eps_coef: epsilon coefficient for approxPolyDP

    Returns:
        list of cropped BGR images (as numpy arrays)
    """
    if frame is None:
        return []
    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # use adaptive threshold for lighting robustness
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # morphology to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps_coef * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype('float32')
            warped = four_point_transform(orig, pts)
            # optionally resize to square
            h, w = warped.shape[:2]
            side = max(h, w)
            square = np.zeros((side, side, 3), dtype=warped.dtype)
            square[:h, :w] = warped
            crops.append(square)
    return crops


def detect_and_crop_from_path(path: str, **kwargs) -> List[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return []
    return detect_and_crop(img, **kwargs)
