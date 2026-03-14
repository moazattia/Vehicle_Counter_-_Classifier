from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# auto-create on import
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Model ──────────────────────────────────────────────
MODEL_SIZE = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.3

# ── Detection ─────────────────────────────────────────
LINE_POSITION = 0.5  # fraction of frame height (0.0 = top, 1.0 = bottom)

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# ── API ───────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = {".mp4", ".avi"}
FRAME_SKIP = 2  # process every Nth frame (1 = all frames, 2 = every other, etc.)
