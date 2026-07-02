import cv2
import pytesseract

image = cv2.imread("test_image.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)

print("Detected text:")
print("----------------")
print(text)
