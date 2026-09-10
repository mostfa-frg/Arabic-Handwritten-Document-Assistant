import os

from dotenv import load_dotenv


load_dotenv(override=True)

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip("\"'") or None
# Change 1: use Groq's supported Qwen model naming while retaining the
# GROQ_MODEL environment override for deployments that choose another model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

TEMPERATURE = 0.1
# Change 1: allow answers with several cited sources to finish without
# truncation.
MAX_TOKENS = 2048
TOP_K = 5
