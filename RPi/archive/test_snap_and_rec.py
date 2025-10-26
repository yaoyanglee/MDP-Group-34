#!/usr/bin/env python3
"""
Standalone test file for the snap_and_rec function.
This allows you to test image capture and recognition without running the full RaspberryPi system.
"""
import json
import time
import os
import requests
import logging
import cv2  
from picamera2 import Picamera2, Preview
from consts import SYMBOL_MAP
from settings import API_IP, API_PORT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(config)
picam2.start()
time.sleep(2)  # let AE stabilize


def snap_and_rec(obstacle_id_with_signal: str) -> dict:
    """
    Simplified version: RPi snaps an image and calls the API for image-rec.
    The response is then returned as a dictionary.
    
    :param obstacle_id_with_signal: the current obstacle ID followed by underscore followed by signal
    :return: Dictionary with recognition results
    """
    obstacle_id, signal = obstacle_id_with_signal.split("_")
    logger.info(f"Capturing image for obstacle id: {obstacle_id}")
    
    url = f"http://{API_IP}:{API_PORT}/snap_image"
    filename = f"{int(time.time())}_{obstacle_id}_{signal}.jpg"

    # Simplified camera capture - just basic libcamera command
    logger.info("Capturing image with basic camera settings...")
    frame = picam2.capture_array()
        
    # Encode to JPEG in-memory
    cv2.imwrite(filename, frame)
    
    if frame is None:
        logger.error("JPEG encoding failed")
    
    
    # Check if file was created
    if not os.path.exists(filename):
        logger.error(f"Image file {filename} was not created")
        return {"error": "Image file not created", "image_id": "NA"}

    logger.debug(f"Image captured: {filename}")
    logger.debug("Requesting from image API")

    # Send image to inference server
    try:
        with open(filename, 'rb') as image_file:
            response = requests.post(
                url, files={"file": (filename, image_file)})

        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}")
            return {"error": f"API returned status {response.status_code}", "image_id": "NA"}

        results = json.loads(response.content)
        logger.info(f"Detection results: {results}")
        
        # Log the detected symbol
        detected_symbol = SYMBOL_MAP.get(results['image_id'], results['image_id'])
        logger.info(f"Detected symbol: {detected_symbol}")
        
        # Add obstacle_id to results for convenience
        results['obstacle_id'] = obstacle_id
        results['signal'] = signal
        results['filename'] = filename
        
        return results
        
    except Exception as e:
        logger.error(f"Error during image capture/recognition: {e}")
        return {"error": str(e), "image_id": "NA"}
    finally:
        # Clean up - remove the captured image file
        # Comment this out if you want to keep the images for inspection
        if os.path.exists(filename):
            logger.info(f"Cleaning up image file: {filename}")
            os.remove(filename)
            # Uncomment the line below if you want to keep the images
            # logger.info(f"Image saved: {filename}")


def test_single_capture(obstacle_id="1", signal="L"):
    """
    Test a single image capture and recognition.
    
    :param obstacle_id: The obstacle ID (default: "1")
    :param signal: The signal direction (default: "left")
    """
    logger.info("=" * 60)
    logger.info(f"Testing snap_and_rec with obstacle_id={obstacle_id}, signal={signal}")
    logger.info("=" * 60)
    
    # Check if API is available
    logger.info(f"Checking API at {API_IP}:{API_PORT}...")
    try:
        status_url = f"http://{API_IP}:{API_PORT}/status"
        response = requests.get(status_url, timeout=2)
        if response.status_code == 200:
            logger.info("✓ API is up and running!")
        else:
            logger.warning(f"⚠ API returned status {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Cannot reach API: {e}")
        logger.error("Make sure the image_api.py server is running!")
        return None
    
    # Run the snap_and_rec function
    obstacle_id_with_signal = f"{obstacle_id}_{signal}"
    results = snap_and_rec(obstacle_id_with_signal)
    
    # Display results
    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info(f"  Obstacle ID: {results.get('obstacle_id', 'N/A')}")
    logger.info(f"  Signal: {results.get('signal', 'N/A')}")
    logger.info(f"  Detected Image ID: {results.get('image_id', 'N/A')}")
    
    if 'error' in results:
        logger.error(f"  Error: {results['error']}")
    else:
        detected = results.get('image_id', 'NA')
        symbol = SYMBOL_MAP.get(detected, detected)
        logger.info(f"  Detected Symbol: {symbol}")
        
        if 'detections' in results:
            logger.info(f"  Number of detections: {len(results['detections'])}")
            for i, det in enumerate(results['detections'], 1):
                logger.info(f"    Detection {i}: {det['class']} (conf: {det['conf']:.2f})")
    
    logger.info("=" * 60)
    
    return results


def test_multiple_captures(obstacle_ids=None, delay=2):
    """
    Test multiple captures in sequence.
    
    :param obstacle_ids: List of (obstacle_id, signal) tuples. Default: [("1", "left"), ("2", "right")]
    :param delay: Delay in seconds between captures (default: 2)
    """
    if obstacle_ids is None:
        obstacle_ids = [("1", "left"), ("2", "right"), ("3", "left")]
    
    logger.info(f"Testing {len(obstacle_ids)} captures with {delay}s delay between them")
    
    results_list = []
    for obstacle_id, signal in obstacle_ids:
        result = test_single_capture(obstacle_id, signal)
        results_list.append(result)
        
        if len(obstacle_ids) > 1:
            logger.info(f"Waiting {delay} seconds before next capture...")
            time.sleep(delay)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY OF ALL CAPTURES:")
    for i, result in enumerate(results_list, 1):
        if result:
            status = "✓ SUCCESS" if 'error' not in result else "✗ FAILED"
            image_id = result.get('image_id', 'N/A')
            logger.info(f"  Capture {i}: {status} - {image_id}")
    logger.info("=" * 60)
    
    return results_list


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("SNAP AND REC TEST UTILITY")
    print("=" * 60)
    print(f"API Configuration: {API_IP}:{API_PORT}")
    print("=" * 60 + "\n")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "multi":
            # Test multiple captures
            test_multiple_captures()
        else:
            # Single capture with custom obstacle_id and signal
            obstacle_id = sys.argv[1] if len(sys.argv) > 1 else "1"
            signal = sys.argv[2] if len(sys.argv) > 2 else "left"
            test_single_capture(obstacle_id, signal)
    else:
        # Default: single capture test
        print("Usage:")
        print("  python test_snap_and_rec.py [obstacle_id] [signal]")
        print("  python test_snap_and_rec.py multi  # Test multiple captures")
        print("\nRunning default test with obstacle_id=1, signal=left\n")
        test_single_capture("1", "left")

