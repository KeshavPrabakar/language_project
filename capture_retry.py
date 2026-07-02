"""Capture multiple frames on user prompt, run OCR variants, aggregate results.
Run:
  python3 capture_retry.py --device 0 --frames 5

The script will prompt you to position the text and press Enter; it captures multiple frames
and prints the most common OCR result (voting across variants and frames).
"""
import argparse
import time
import os
from collections import Counter

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--outdir', default='capture_retry_out')
    p.add_argument('--delay', type=float, default=0.25)
    return p.parse_args()


def ensure(d):
    if not os.path.exists(d):
        os.makedirs(d)


def make_variants(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variants = []
    variants.append(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    variants.append(clahe_img)

    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    adaptive = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    variants.append(adaptive)

    sharpen = cv2.filter2D(clahe_img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    _, sharp = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(sharp)

    return variants


def ocr_from_image(img):
    # return assembled text string (joining words in reading order)
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception:
        try:
            return pytesseract.image_to_string(img).strip(), 0.0
        except Exception:
            return '', 0.0

    texts = []
    confs = []
    n = len(data.get('text', []))
    for i in range(n):
        t = str(data.get('text', [])[i] or '').strip()
        if not t:
            continue
        try:
            conf = float(data.get('conf', [])[i])
        except Exception:
            conf = 0.0
        texts.append((data.get('left', [])[i], t, conf))
        confs.append(conf)

    if not texts:
        return '', 0.0
    texts.sort(key=lambda x: x[0])
    joined = ' '.join(t for _, t, _ in texts)
    avg_conf = sum([c for c in confs if c >= 0]) / max(1, len([c for c in confs if c >= 0]))
    return joined.strip(), avg_conf


def main():
    args = parse_args()
    ensure(args.outdir)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print('Could not open camera', args.device)
        return

    input('Position the printed text "hello this is a test" in front of the camera, then press Enter to capture frames...')
    captured = []
    for i in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            print('Frame capture failed')
            break
        path = os.path.join(args.outdir, f'frame_{i}.jpg')
        cv2.imwrite(path, frame)
        captured.append((frame, path))
        time.sleep(args.delay)

    cap.release()
    print(f'Captured {len(captured)} frames; running OCR variants...')

    votes = Counter()
    details = []
    for idx, (frame, path) in enumerate(captured):
        variants = make_variants(frame)
        for v_index, var in enumerate(variants):
            text, conf = ocr_from_image(var)
            if text:
                key = text.strip()
                votes[key] += 1
                details.append((path, v_index, key, conf))

    if not votes:
        print('No text detected in any frame/variant. Ensure the text is clear, large, and well-lit.')
        return

    best_text, count = votes.most_common(1)[0]
    print('\nMost common OCR result (voted across frames/variants):')
    print('"' + best_text + '"', f'(votes: {count})')
    print('\nAll detections (path, variant-index, text, avg-conf):')
    for d in details:
        print(d)

if __name__ == '__main__':
    main()
