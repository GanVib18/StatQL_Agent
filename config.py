import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GROQ_MODEL:   str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DB_PATH:      str = os.getenv("DB_PATH",    "retail.duckdb")
CACHE_META:   str = os.getenv("CACHE_META", "cache_meta.json")
CACHE_INDEX:  str = os.getenv("CACHE_INDEX","cache_index.faiss")
CACHE_THRESH: float = float(os.getenv("CACHE_THRESH", "0.92"))
