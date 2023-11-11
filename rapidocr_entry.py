import cv2
from rapidocr_onnxruntime import RapidOCR

rapid_ocr = RapidOCR()

image_path = './test/test_images/sparse_text_square.jpg'
img = cv2.imread(image_path)
ocr_result, _ = rapid_ocr(img)

for v in ocr_result:
    print(v)