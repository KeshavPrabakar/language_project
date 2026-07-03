# language_project
This is my NVIDIA project. It takes in words in English and can translate it to a different language of your choice from the options including es-Spanish, hi-Hindi, ta-Tamil, zh Chinese, de-German, ja- Japanese.  
                                           video on how it works- https://www.image2url.com/r2/default/videos/1783027703363-f9783d78-12ec-47d0-8933-46a143aef6cf.mp4


The project is a command-line OCR and translation pipeline that starts with an image containing English text, extracts the text using Tesseract, and then translates that text into another language. It keeps the core files image_ocr.py, translate_cli.py, translator.py, and english_hi_dataset.tsv, where image_ocr.py handles OCR from an image and translate_cli.py reads the OCR output and translates it.

For Hindi translation, the project first checks the local dataset file english_hi_dataset.tsv for an exact phrase match and returns the stored Hindi result if available; otherwise it falls back to deep_translator.GoogleTranslator for online translation. This makes the project useful for converting printed English text from images into translated text with a small built-in phrase dataset and an extendable translation
