from picamera2 import Picamera2
import time

# Initialize camera
picam2 = Picamera2()

# Configure camera
config = picam2.create_still_configuration()
picam2.configure(config)

# Start camera
picam2.start()

# Wait for camera to be ready
time.sleep(2)

# Take photo
picam2.capture_file('test.jpg')
print("Photo captured as test.jpg")