#!/usr/bin/env python3
"""Simple CLI to OCR an input image with multiple preprocessing variants.
Usage:
  python3 image_ocr.py --image path/to/img.jpg --lang eng
"""
import argparse
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def parse_args():
    p = argparse.ArgumentParser(description='OCR a single image file')
    p.add_argument('--image', required=True, help='Path to input image')
    p.add_argument('--lang', default='eng', help='Tesseract language code (default: eng)')
    p.add_argument('--psm', default='6', help='Tesseract PSM mode (default 6)')
    return p.parse_args()


def resize_frame(frame, target_width=1200):
    h, w = frame.shape[:2]
    if w > target_width:
        scale = target_width / float(w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def make_variants(frame):
    frame = resize_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    variants = {'gray': gray}

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    variants['clahe'] = clahe_img

    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['otsu'] = otsu

    adaptive = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    variants['adaptive'] = adaptive

    median = cv2.medianBlur(clahe_img, 3)
    variants['median'] = median

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(clahe_img, -1, kernel)
    _, sharp_thr = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['sharp'] = sharp_thr

    inverted = cv2.bitwise_not(otsu)
    variants['inverted'] = inverted

    return variants


def build_text_from_data(data):
    lines = {}
    n = len(data.get('text', []))
    for i in range(n):
        text = str(data.get('text', [])[i] or '').strip()
        if not text:
            continue
        try:
            conf = float(data.get('conf', [])[i])
        except Exception:
            conf = 0.0
        if conf < 5:
            continue
        block = data.get('block_num', [])[i]
        par = data.get('par_num', [])[i]
        line = data.get('line_num', [])[i]
        left = data.get('left', [])[i]
        width = data.get('width', [])[i]
        key = (block, par, line)
        lines.setdefault(key, []).append((left, width, text, conf))

    assembled_lines = []
    confs = []
    for key in sorted(lines.keys()):
        words = sorted(lines[key], key=lambda item: item[0])
        if not words:
            continue
        line_text = ''
        prev_right = None
        avg_char_width = max(1, sum(word[1] for word in words) / max(1, sum(len(word[2]) for word in words)))
        for left, width, word, conf in words:
            if prev_right is None:
                line_text += word
            else:
                gap = left - prev_right
                if gap > avg_char_width * 1.0:
                    line_text += ' '
                line_text += word
            prev_right = left + width
            confs.append(conf)
        assembled_lines.append(line_text.strip())
    return '\n'.join(assembled_lines).strip(), confs


def ocr_on_image(image, lang='eng', psm='6'):
    best_text = ''
    best_score = 0.0
    for name, img in make_variants(image).items():
        try:
            config = f"--oem 3 --psm {psm} -l {lang}"
            data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
            text, confs = build_text_from_data(data)
        except Exception:
            try:
                text = pytesseract.image_to_string(img, lang=lang)
                confs = []
            except Exception:
                text = ''
                confs = []
        score = (sum(confs) / len(confs) * len(confs)) if confs else (len(text) if text else 0)
        if score > best_score and text:
            best_score = score
            best_text = text
    return best_text, best_score


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print('Could not read image:', args.image)
        return
    text, score = ocr_on_image(img, lang=args.lang, psm=args.psm)
    if not text:
        print('<no text detected>')
    else:
        print('Detected text:')
        print('----------------')
        print(text)

if __name__ == '__main__':
    main()
