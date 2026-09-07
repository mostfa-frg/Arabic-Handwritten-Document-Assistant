import os

from dotenv import load_dotenv


load_dotenv(override=True)

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip("\"'") or None
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

TEMPERATURE = 0.1
MAX_TOKENS = 1024
TOP_K = 5
