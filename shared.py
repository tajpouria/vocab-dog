import os
import logging
import sys
from google import genai

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger: logging.Logger = logging.getLogger(name)
    return logger


genai_client = genai.Client()
genai_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
