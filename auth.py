from openai import OpenAI

from config import EMBEDDING_MODEL


def validate_openai_key(api_key: str) -> tuple[bool, str]:
    """
    Validate an OpenAI API key by making a minimal embeddings request.

    Returns:
        (True, success message) when valid.
        (False, safe error message) when invalid.
    """
    if not api_key or not api_key.strip():
        return False, "Please enter an OpenAI API key."

    try:
        client = OpenAI(api_key=api_key.strip())

        client.embeddings.create(
            model=EMBEDDING_MODEL,
            input="API key validation test",
        )

        return True, "API key validated successfully."

    except Exception:
        return False, "The API key could not be validated. Check the key, account access, and internet connection."