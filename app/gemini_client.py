from google import genai

# Initialize the client with your API key
client = genai.Client(api_key="xyz")  # keep this secret!


def build_prompt(raw_text: str) -> str:
    return (
        "You are a Nepali Braille OCR post-processing assistant. "
        "I will give you text that was decoded from Braille (OBR) and may be noisy. "
        "Your task:\n"
        "1) Correct spacing, spelling (if possible), and combine characters into proper Nepali words.\n"
        "2) Do NOT add explanations, notes, or extra words.\n"
        "3) Output ONLY the final cleaned Nepali text. No quotes, no prefixes, no translation.\n\n"
        f"OBR decoded text:\n{raw_text}"
    )

def clean_braille_text(raw_text: str) -> str:
    # if not raw_text.strip():
    #     return raw_text

    response = client.models.generate_content(
        model="gemini-3-flash",
        contents=build_prompt(raw_text)
    )

    return response.text.strip()
    
# Example usage
# raw_text = "obR nEpALi tExT wIth sPacing eRrors"
# clean_text = clean_braille_text(raw_text)
# print("Cleaned text:", clean_text)
