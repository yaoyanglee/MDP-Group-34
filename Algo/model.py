from __future__ import annotations

import glob
import os
import shutil
import time
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# configuration helpers


@dataclass(frozen=True)
class _ModelSource:
    weight_file: str = "Week_9.pt"
    repo_root: str = "./"
    entrypoint: str = "custom"


@dataclass(frozen=True)
class _FolderLayout:
    uploads: str = "uploads"
    runs: str = "runs"
    runs_originals: str = os.path.join(runs, "originals")
    own_results: str = "own_results"


@dataclass(frozen=True)
class _DetectionClassMap:
    name_to_id: Mapping[str, str]

    @staticmethod
    def default() -> "_DetectionClassMap":
        mapping = {
            "NA": "NA",
            "Bullseye": "10",
            "One": "11",
            "Two": "12",
            "Three": "13",
            "Four": "14",
            "Five": "15",
            "Six": "16",
            "Seven": "17",
            "Eight": "18",
            "Nine": "19",
            "A": "20",
            "B": "21",
            "C": "22",
            "D": "23",
            "E": "24",
            "F": "25",
            "G": "26",
            "H": "27",
            "S": "28",
            "T": "29",
            "U": "30",
            "V": "31",
            "W": "32",
            "X": "33",
            "Y": "34",
            "Z": "35",
            "Up": "36",
            "Down": "37",
            "Right": "38",
            "Left": "39",
            "Up Arrow": "36",
            "Down Arrow": "37",
            "Right Arrow": "38",
            "Left Arrow": "39",
            "Stop": "40",
        }
        return _DetectionClassMap(mapping)

    def resolve(self, label: str) -> str:
        return self.name_to_id[label]


SOURCE = _ModelSource()
FOLDERS = _FolderLayout()
CLASS_IDS = _DetectionClassMap.default()


# ---------------------------------------------------------------------------
# public API


def load_model():
    """Load the YOLOv5 weights from disk via torch hub."""

    return torch.hub.load(
        SOURCE.repo_root,
        SOURCE.entrypoint,
        path=SOURCE.weight_file,
        source="local",
    )


def predict_image(filename: str, model, signal: str) -> str:
    """Run inference on an uploaded image and return the chosen class ID string."""

    try:
        image = _open_upload(filename)
        result_bundle = _run_inference(model, image)
        predictions = _prepare_predictions(result_bundle)
        filtered = _filter_candidates(predictions, signal)
        _draw_if_needed(image, filtered)
        label = _extract_identifier(filtered)
        print(f"Final result: {label}")
        return label
    except Exception:
        print("Final result: NA")
        return "NA"


def predict_image_week_9(filename: str, model) -> str:
    image = _open_upload(filename)
    result_bundle = _run_inference(model, image)
    predictions = _prepare_predictions(result_bundle)
    chosen = _pick_first_valid(predictions)
    _draw_if_needed(image, chosen)
    return _extract_identifier(chosen)


def stitch_image():
    return _stitch_detect_folder(FOLDERS.runs, move_originals=True)


def stitch_image_own():
    return _stitch_custom_results()


# ---------------------------------------------------------------------------
# inference helpers


def _open_upload(filename: str) -> Image.Image:
    path = os.path.join(FOLDERS.uploads, filename)
    return Image.open(path)


def _run_inference(model, image: Image.Image):
    results = model(image)
    results.save(FOLDERS.runs)
    return results


def _prepare_predictions(results) -> List[Mapping]:
    dataframe = results.pandas().xyxy[0].copy()
    dataframe["bboxHt"] = dataframe["ymax"] - dataframe["ymin"]
    dataframe["bboxWt"] = dataframe["xmax"] - dataframe["xmin"]
    dataframe["bboxArea"] = dataframe["bboxHt"] * dataframe["bboxWt"]
    dataframe = dataframe.sort_values("bboxArea", ascending=False)
    return dataframe.to_dict("records")


def _filter_candidates(predictions: Sequence[Mapping], signal: str):
    pool = [p for p in predictions if p["name"] != "Bullseye"]
    if not pool:
        return "NA"

    if len(pool) == 1:
        return pool[0]

    shortlist: List[Mapping] = []
    current_area = pool[0]["bboxArea"]
    for candidate in pool:
        is_one = candidate["name"] == "One"
        threshold = 0.6 if is_one else 0.8
        if (
            candidate["confidence"] > 0.5
            and candidate["bboxArea"] >= current_area * threshold
        ):
            shortlist.append(candidate)
            current_area = candidate["bboxArea"]

    if len(shortlist) == 1:
        return shortlist[0]

    if not shortlist:
        return pool[0]

    shortlist.sort(key=lambda item: item["xmin"])
    if signal == "L":
        return shortlist[0]
    if signal == "R":
        return shortlist[-1]

    for candidate in shortlist:
        if 250 < candidate["xmin"] < 774:
            return candidate

    shortlist.sort(key=lambda item: item["bboxArea"])
    return shortlist[-1]


def _pick_first_valid(predictions: Sequence[Mapping]):
    for candidate in predictions:
        if candidate["name"] != "Bullseye" and candidate["confidence"] > 0.5:
            return candidate
    return "NA"


def _draw_if_needed(image: Image.Image, candidate):
    if isinstance(candidate, str):
        return
    _draw_bounding_box(np.array(image), candidate)


def _draw_bounding_box(raw_image: np.ndarray, candidate: Mapping) -> None:
    label = candidate["name"]
    identifier = CLASS_IDS.resolve(label)
    composite = f"{label}-{identifier}"

    x1, y1, x2, y2 = (
        int(candidate["xmin"]),
        int(candidate["ymin"]),
        int(candidate["xmax"]),
        int(candidate["ymax"]),
    )

    timestamp = str(int(time.time()))
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    _write_image(os.path.join(FOLDERS.own_results, f"raw_image_{composite}_{timestamp}.jpg"), rgb_image)

    annotated = cv2.rectangle(rgb_image.copy(), (x1, y1), (x2, y2), (36, 255, 12), 2)
    (width, _), baseline = cv2.getTextSize(composite, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(annotated, (x1, y1 - 20), (x1 + width, y1), (36, 255, 12), -1)
    cv2.putText(annotated, composite, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    _write_image(
        os.path.join(FOLDERS.own_results, f"annotated_image_{composite}_{timestamp}.jpg"),
        annotated,
    )


def _write_image(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def _extract_identifier(candidate) -> str:
    if isinstance(candidate, str):
        return candidate
    return CLASS_IDS.resolve(candidate["name"])


# ---------------------------------------------------------------------------
# stitching helpers


def _stitch_detect_folder(folder: str, move_originals: bool) -> Image.Image:
    destination = os.path.join(folder, f"stitched-{int(time.time())}.jpeg")
    pattern = os.path.join(folder, "detect/*/*.jpg")
    images = _load_images(glob.glob(pattern))

    stitched = _concatenate_horizontally(images)
    stitched.save(destination)

    if move_originals:
        os.makedirs(FOLDERS.runs_originals, exist_ok=True)
        for path in glob.glob(pattern):
            shutil.move(path, os.path.join(FOLDERS.runs_originals, os.path.basename(path)))

    return stitched


def _stitch_custom_results() -> Image.Image:
    destination = os.path.join(FOLDERS.own_results, f"stitched-{int(time.time())}.jpeg")
    pattern = os.path.join(FOLDERS.own_results, "annotated_image_*.jpg")
    pairs = _sorted_by_timestamp(glob.glob(pattern))
    images = _load_images([path for path, _ in pairs])
    stitched = _concatenate_horizontally(images)
    stitched.save(destination)
    return stitched


def _sorted_by_timestamp(paths: Iterable[str]) -> List[tuple[str, str]]:
    def _timestamp(path: str) -> str:
        return path.split("_")[-1].split(".")[0]

    return sorted(((path, _timestamp(path)) for path in paths), key=lambda item: item[1])


def _load_images(paths: Iterable[str]) -> List[Image.Image]:
    return [Image.open(path) for path in paths]


def _concatenate_horizontally(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images available for stitching")

    widths, heights = zip(*(image.size for image in images))
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.size[0]
    return canvas


__all__ = [
    "load_model",
    "predict_image",
    "predict_image_week_9",
    "stitch_image",
    "stitch_image_own",
]

