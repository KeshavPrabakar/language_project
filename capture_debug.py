"""Capture one frame from camera and run OCR with several preprocessing variants.
Usage:
  python3 capture_debug.py --device 0 --out debug.jpg

The script prints OCR outputs for each variant and saves the variant images to ./debug_variants/.
"""
import argparse
import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", default="capture.jpg")
    p.add_argument("--dir", default="debug_variants")
    return p.parse_args()


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def make_variants(img):
    # img: BGR
    frame = img.copy()
    variants = {}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variants['gray'] = gray

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    variants['clahe'] = clahe_img

    # Otsu
    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['otsu'] = otsu

    # Adaptive
    adaptive = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    variants['adaptive'] = adaptive

    # Median blur
    median = cv2.medianBlur(clahe_img, 3)
    variants['median'] = median

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(clahe_img, -1, kernel)
    variants['sharp'] = sharp

    # Inverted
    inverted = cv2.bitwise_not(otsu)
    variants['inverted'] = inverted

    return variants


def ocr_variant(img):
    # prefer image_to_data to compute average confidence
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception as e:
        return "", 0.0
    texts = []
    confs = []
    for t, c in zip(data.get('text', []), data.get('conf', [])):
        if not t or not str(t).strip():
            continue
        try:
            conf = float(c)
        except Exception:
            conf = -1
        texts.append(str(t).strip())
        confs.append(conf)
    joined = ' '.join(texts).strip()
    avg_conf = sum([c for c in confs if c >= 0]) / max(1, len([c for c in confs if c >= 0])) if confs else 0.0
    return joined, avg_conf


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print('Could not open camera', args.device)
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('Could not capture frame')
        return

    cv2.imwrite(args.out, frame)
    print('Saved capture to', args.out)

    ensure_dir(args.dir)
    variants = make_variants(frame)

    results = []
    for name, img in variants.items():
        path = os.path.join(args.dir, f"{name}.png")
        # write grayscale images correctly
        if len(img.shape) == 2:
            cv2.imwrite(path, img)
        else:
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        text, conf = ocr_variant(img)
        results.append((name, text, conf, path))

    print('\nOCR results for capture:')
    for name, text, conf, path in sorted(results, key=lambda r: -r[2]):
        print('---')
        print(f'Variant: {name}  (avg conf: {conf:.1f})')
        print(text if text else '<no text detected>')
        print('saved to', path)

if __name__ == "__main__":
    main()
