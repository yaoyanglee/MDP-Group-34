#!/usr/bin/env python3
import json
import time
import threading
import requests
#from communication.android import AndroidLink, AndroidMessage
from communication.stm32 import STMLink
from logger import prepare_logger
from settings import API_IP, API_PORT
from picamera2 import Picamera2
import cv2

logger = prepare_logger()

def camera_loop():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640,480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # let exposure settle

    url = f"http://{API_IP}:{API_PORT}/image"

    try:
        while True:
            frame = picam2.capture_array()
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                logger.error("JPEG encoding failed")
                continue

            try:
                response = requests.post(url, files={"file": ("frame.jpg", jpeg.tobytes())}, timeout=2)
                if response.status_code != 200:
                    logger.error(f"Server error: {response.status_code}")
            except Exception as e:
                logger.error(f"HTTP error: {e}")

            time.sleep(0.1)  # ~10 FPS
    finally:
        picam2.stop()


class RaspberryPi:
    """
    Simplified Raspberry Pi orchestrator for start → detect → STM control flow.
    """

    def __init__(self):
        self.logger = prepare_logger()
        #self.android_link = AndroidLink()
        self.stm_link = STMLink()
        self.running = False
        self.start_time = None

    def recv_nonblocking(self):
        """Return STM message if available, else None"""
        try:
            # use select or settimeout depending on implementation
            self.conn.settimeout(0.01)   # very short timeout
            return self.recv()
        except Exception:
            return None


    def start(self):
        """Starts the RPi orchestrator."""
        try:
            #self.android_link.connect()
            self.stm_link.connect()
            #self.android_link.send(AndroidMessage("info", "Connected to RPi! Ready."))

            #self.logger.info("Listening for Android commands...")
            #self.listen_android()
            self.wait_terminal()
            self.run_script_flow()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Gracefully disconnect and exit."""
        self.logger.info("Shutting down...")
        #self.android_link.disconnect()
        self.stm_link.disconnect()
        self.logger.info("Program exited!")

    def listen_android(self):
        """Listen for messages from Android."""
        while True:
            msg_str = self.android_link.recv()
            if not msg_str:
                continue

            message = json.loads(msg_str)
            if message["cat"] == "control" and message["value"] == "start":
                self.logger.info("Received START command from Android.")
                self.android_link.send(AndroidMessage("info", "Starting script..."))
                self.run_script_flow()

    def wait_terminal(self):
        try:
            while True:
                user = input("Type 'S' and press Enter to start the script: ").strip().lower()
                if user == 's':
                    self.logger.info("Terminal START received.")
                    break
                else:
                    self.logger.info("Ignored input; waiting for 'S'...")
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt during wait; exiting.")
            raise

    def run_script_flow(self):
        """
        Flow:
        1. Send 'S' to STM
        2. Poll detection API every 2s
        3. Send 'L'/'R' continuously based on detection
        4. Stop only if STM32 says SCRIPT_DONE or timeout (3min)
        """
        self.running = True
        self.start_time = time.time()
        timeout = 180  # seconds

        self.logger.info("Sending 'S' to STM32...")
        self.stm_link.send("S")
        time.sleep(2)
        self.logger.info("Begin detection loop.")

        last_cmd = None  # remember what we last sent, to avoid spam if detection stays same

        while self.running:
            # Timeout guard
            if time.time() - self.start_time > timeout:
                self.logger.warning("Timeout reached (3 minutes). Exiting loop.")
                #self.android_link.send(AndroidMessage("error", "Timeout reached. Exiting."))
                break

            # msg = self.stm_link.recv_nonblocking()
            # if msg:
            #     self.logger.debug(f"Received from STM32: {msg}")
            #     if "SCRIPT_DONE" in msg:
            #         self.logger.info("SCRIPT_DONE received.")
            #         self.android_link.send(AndroidMessage("info", "SCRIPT_DONE received."))
            #         break

            # ---- GET DETECTION ----
            detection = self.get_detection()
            if detection:
                self.logger.info(f"Detection: {detection}")
                # only send if new or every 2 seconds anyway
                if detection == "left":
                    self.stm_link.send("L")
                    self.logger.debug("Sent 'L' to STM32.")
                    last_cmd = "L"
                elif detection == "right":
                    self.stm_link.send("R")
                    self.logger.debug("Sent 'R' to STM32.")
                    last_cmd = "R"
                else:
                    self.logger.debug(f"Ignored unknown detection: {detection}")
            else:
                self.logger.debug("No detection result received (continuing).")

            # wait 2s before next query regardless of detection outcome
            time.sleep(2)

        self.running = False
        #self.android_link.send(AndroidMessage("status", "finished"))
        self.logger.info("Script finished.")


    def get_detection(self):
        """Query the image recognition API for current detections."""
        url = f"http://{API_IP}:{API_PORT}/latest_detections"
        try:
            response = requests.get(url, timeout=2)
            if response.status_code != 200:
                self.logger.error(f"Detection API error {response.status_code}")
                return None

            data = response.json()
            detections = data.get("detections", [])
            if not detections:
                return None

            # choose highest-confidence detection
            top = max(detections, key=lambda d: d.get("conf", 0))
            cls = top.get("class", "").lower()

            if "left" in cls:
                return "left"
            elif "right" in cls:
                return "right"
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error calling detection API: {e}")
            return None



if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    rpi = RaspberryPi()
    rpi.start()
