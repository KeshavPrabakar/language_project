from deep_translator import GoogleTranslator


def translate_text(text: str, target_language: str, source_language: str = 'auto') -> str:
    """Translate text into the target language using deep_translator."""
    translator = GoogleTranslator(source=source_language, target=target_language)
    return translator.translate(text)
