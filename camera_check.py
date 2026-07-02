import cv2

camera = cv2.VideoCapture(0)

if camera.isOpened():
    print("Camera detected!")
else:
    print("No camera found")

camera.release()

