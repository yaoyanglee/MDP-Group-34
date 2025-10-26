#!/usr/bin/env python3
import time
import requests
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from logger import prepare_logger
from settings import API_IP, API_PORT
from consts import SYMBOL_MAP

logger = prepare_logger()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(config)
picam2.start()
time.sleep(2)  # let AE stabilize

url = f"http://{API_IP}:{API_PORT}/image"

global detection_list

try:
    while True:
        # Capture frame as array
        frame = picam2.capture_array()
        # Encode to JPEG in-memory
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            logger.error("JPEG encoding failed")
            continue

        try:
            response = requests.post(url, files={"file": ("frame.jpg", jpeg.tobytes())}, timeout=2)
            if response.status_code == 200:
                data = response.json()
                image_id = data.get("image_id", "NA")
                detections = data.get("detections", [])
                detection_list = detections
                logger.info(f"Class: {image_id}, Detections: {len(detections)}")
            else:
                logger.error(f"Server error: {response.status_code}")
        except Exception as e:
            logger.error(f"HTTP error: {e}")

        time.sleep(0.1)  # ~10 FPS

finally:
    picam2.stop()


