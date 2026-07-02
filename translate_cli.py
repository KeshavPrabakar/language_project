#!/usr/bin/env python3
"""Translate English text into chosen target language(s).
Supported targets: es (Spanish), ta (Tamil), hi (Hindi), zh (Mandarin), de (German), ja (Japanese)

Usage:
  python3 translate_cli.py --text "hello this is a test" --to es
  python3 translate_cli.py --file input.txt --to hi
  python3 translate_cli.py --all  # translate to all supported languages
"""
import argparse
import os
from deep_translator import GoogleTranslator

SUPPORTED = {
    'es': 'Spanish',
    'ta': 'Tamil',
    'hi': 'Hindi',
    'zh': 'Chinese (Simplified)',
    'zh-CN': 'Chinese (Simplified)',
    'zh-TW': 'Chinese (Traditional)',
    'de': 'German',
    'ja': 'Japanese'
}

TARGET_ALIASES = {
    'zh': 'zh-CN'
}

HI_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'english_hi_dataset.tsv')

TARGET_ALIASES = {
    'zh': 'zh-CN'
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--text', help='Text to translate (English)')
    p.add_argument('--file', help='Path to text file with English text')
    p.add_argument('--to', help='Target language code (es, ta, hi, zh, zh-CN, zh-TW, de, ja)')
    p.add_argument('--all', action='store_true', help='Translate to all supported languages')
    return p.parse_args()


def load_hi_dataset():
    dataset = {}
    if not os.path.exists(HI_DATASET_PATH):
        return dataset
    try:
        with open(HI_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                english, hindi = line.rstrip('\n').split('\t', 1)
                dataset[english.strip().lower()] = hindi.strip()
    except Exception:
        pass
    return dataset


def translate(text, target):
    if target == 'hi':
        dataset = load_hi_dataset()
        if text.strip().lower() in dataset:
            return dataset[text.strip().lower()]
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        return f'<translation failed: {e}>'


def main():
    args = parse_args()
    if not args.text and not args.file:
        print('Provide --text or --file')
        return
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"File not found: {args.file}")
            return
        except Exception as e:
            print(f"Failed to read file {args.file}: {e}")
            return
    else:
        text = args.text.strip()

    if not text:
        print('No text to translate')
        return

    targets = []
    if args.all:
        targets = list(SUPPORTED.keys())
    elif args.to:
        normalized = TARGET_ALIASES.get(args.to, args.to)
        if normalized not in SUPPORTED:
            print('Unsupported language. Supported codes:', ','.join(SUPPORTED.keys()))
            return
        targets = [normalized]
    else:
        print('Specify --to or --all')
        return

    for t in targets:
        lang_name = SUPPORTED.get(t, t)
        translated = translate(text, t)
        print(f'--- {lang_name} ({t}) ---')
        print(translated)
        print()

if __name__ == '__main__':
    main()
