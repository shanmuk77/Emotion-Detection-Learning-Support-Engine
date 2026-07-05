from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

HISTORY_CSV = DATA_DIR / "learning_history.csv"

# Load environment variables
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMOTIONS = ["Bored", "Frustrated", "Confident", "Confused", "Curious"]
MIXED_THRESHOLD = 0.20
MAX_LEN = 100
