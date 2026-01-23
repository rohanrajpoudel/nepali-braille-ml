import logging
from typing import Optional

import requests


logger = logging.getLogger(__name__)


# NOTE: For testing only. Do NOT commit real keys in production.
GEMINI_API_KEY = "AIzaSyBQDocGZolRPLkK8qxDz84rjsQAyrTGV9w"
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


def _build_prompt(raw_text: str) -> str:
    """
    Build the instruction prompt for Gemini.
    The goal is to clean/normalize the OBR Braille Nepali text into
    proper words/sentences and return ONLY the final clean text.
    """
    instruction = (
        "You are a Nepali Braille OCR post-processing assistant. "
        "I will give you text that was decoded from Braille (OBR) and may be noisy. "
        "Your task:\n"
        "1) Correct spacing, spelling (if possible), and combine characters into proper Nepali words.\n"
        "2) Do NOT add explanations, notes, or extra words.\n"
        "3) Output ONLY the final cleaned Nepali text. No quotes, no prefixes, no translation.\n\n"
        "OBR decoded text:\n"
    )
    return instruction + raw_text


def call_gemini_clean_text(raw_text: str) -> Optional[str]:
    """
    Call Gemini API to clean / normalize the decoded Braille Nepali text.

    Returns:
        Cleaned text string, or None if call fails.
    """

    if not raw_text or not raw_text.strip():
        return raw_text

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_prompt(raw_text)
                    }
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            params={"key": api_key},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning("Gemini response has no candidates.")
            return None

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            logger.warning("Gemini response has no parts.")
            return None

        text = parts[0].get("text", "").strip()
        return text or None

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return None

