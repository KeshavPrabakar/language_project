
import argparse
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from translator import translate_text


def parse_args():
    parser = argparse.ArgumentParser(description="Live camera OCR and translation")
    parser.add_argument("--target", default="en", help="Target language code for translation (e.g. en, es, fr, de, zh)")
    parser.add_argument("--source", default="auto", help="Source language code for OCR/translation")
    parser.add_argument("--device", default=0, type=int, help="Camera device index")
    parser.add_argument("--skip", default=8, type=int, help="Frames to skip between OCR attempts")
    parser.add_argument("--confidence", default=80.0, type=float, help="Minimum score to accept OCR text")
    return parser.parse_args()


def resize_frame(frame):
    height, width = frame.shape[:2]
    if width < 900:
        scale = 900.0 / width
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif width > 1600:
        scale = 1600.0 / width
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def crop_center(frame, ratio=0.9):
    h, w = frame.shape[:2]
    mh = int((1 - ratio) * h / 2)
    mw = int((1 - ratio) * w / 2)
    return frame[mh : h - mh, mw : w - mw]


def detect_text_rois(gray):
    grad = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    connected = cv2.dilate(connected, kernel, iterations=2)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    h, w = gray.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 80 or ch < 20 or cw * ch < 1500:
            continue
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        rois.append(gray[y0:y1, x0:x1])
    return rois[:3]


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def make_variants(frame):
    frame = resize_frame(frame)
    frame = crop_center(frame, ratio=0.95)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    median = cv2.medianBlur(gray, 3)
    sharpened = sharpen(gray)
    _, sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(otsu)

    variants = [gray, otsu, adaptive, sharp, inverted, median]
    for roi in detect_text_rois(gray):
        if roi.size == 0:
            continue
        variants.append(roi)
        _, roi_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(roi_otsu)
    return variants


def build_text_from_data(data):
    lines = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data.get("text", [])[i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data.get("conf", [])[i])
        except (ValueError, IndexError):
            continue
        if conf < 10:
            continue

        block = data.get("block_num", [])[i]
        par = data.get("par_num", [])[i]
        line = data.get("line_num", [])[i]
        left = data.get("left", [])[i]
        width = data.get("width", [])[i]
        key = (block, par, line)
        lines.setdefault(key, []).append((left, width, text, conf))

    assembled_lines = []
    confs = []
    for key in sorted(lines.keys()):
        words = sorted(lines[key], key=lambda item: item[0])
        if not words:
            continue

        line_text = ""
        prev_right = None
        avg_char_width = max(1, sum(word[1] for word in words) / max(1, sum(len(word[2]) for word in words)))

        for left, width, word, conf in words:
            if prev_right is None:
                line_text += word
            else:
                gap = left - prev_right
                if gap > avg_char_width * 1.0:
                    line_text += " "
                line_text += word
            prev_right = left + width
            confs.append(conf)

        assembled_lines.append(line_text.strip())

    return "\n".join(assembled_lines).strip(), confs


def ocr_score(confs):
    if not confs:
        return 0.0
    avg_conf = sum(confs) / len(confs)
    return avg_conf * len(confs)


def ocr_on_image(image):
    best_text = ""
    best_score = 0.0

    for psm in [6, 4, 3, 11]:
        config = f"--oem 3 --psm {psm} -l eng"
        data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
        text, confs = build_text_from_data(data)
        if not text:
            continue

        score = ocr_score(confs)
        if score > best_score and len(text) >= len(best_text):
            best_text = text
            best_score = score

    return best_text, best_score


def translate_text_if_needed(text, source, target):
    if not text:
        return None
    return translate_text(text, source_language=source, target_language=target)


def main():
    args = parse_args()
    camera = cv2.VideoCapture(args.device)
    if not camera.isOpened():
        print("Could not open camera.")
        return

    print("Starting live camera OCR and translation.")
    print("Show text to the camera and press Ctrl+C to stop.")
    frame_counter = 0
    last_text = ""

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Could not read frame.")
                break

            frame_counter += 1
            if frame_counter % args.skip != 0:
                continue

            candidates = make_variants(frame)
            best_text = ""
            best_score = 0.0

            for candidate in candidates:
                text, score = ocr_on_image(candidate)
                if score > best_score:
                    best_score = score
                    best_text = text

            if best_score < args.confidence or not best_text:
                last_text = ""
                continue

            if best_text != last_text:
                print("Detected text:")
                print("----------------")
                print(best_text)
                if args.target != "en":
                    translation = translate_text_if_needed(best_text, args.source, args.target)
                    if translation:
                        print(f"Translated ({args.target}):")
                        print(translation)
                print()
                last_text = best_text
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
