import cv2
import time

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

print("Taking picture in 5 seconds...")
time.sleep(5)

ret, frame = camera.read()

if ret:
    cv2.imwrite("test_image.jpg", frame)
    print("Image saved!")
else:
    print("Could not capture image")

camera.release()

