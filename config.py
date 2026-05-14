import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ─────────────────────────────────────────────
# Folder Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR      = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ─────────────────────────────────────────────
# Model Names
# ─────────────────────────────────────────────
VISION_MODEL     = "gemini-2.5-flash"   # Used for image description
CHAT_MODEL       = "gemini-2.5-flash"   # Used for answer generation
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # HF embeddings

# ─────────────────────────────────────────────
# Chunking Settings
# ─────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# ─────────────────────────────────────────────
# Semantic Chunking Settings
# ─────────────────────────────────────────────
SEMANTIC_BREAKPOINT_TYPE       = "percentile"  # Options: percentile, standard_deviation, interquartile
SEMANTIC_BREAKPOINT_THRESHOLD  = 85            # Higher = fewer, larger chunks. Lower = more, smaller chunks
MAX_SEMANTIC_CHUNK_SIZE        = 1500          # Hard cap — semantic chunks larger than this get split further
MIN_SEMANTIC_CHUNK_SIZE        = 50            # Hard floor — chunks smaller than this get dropped

# ─────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────
TOP_K_RESULTS = 6

# ─────────────────────────────────────────────
# Confidence Score Thresholds
# ─────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.60
CONFIDENCE_MEDIUM = 0.40

# ─────────────────────────────────────────────
# Supported File Types
# ─────────────────────────────────────────────
PDF_EXTENSIONS       = [".pdf"]
IMAGE_EXTENSIONS     = [".jpg", ".jpeg", ".png", ".webp"]
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS + IMAGE_EXTENSIONS

# ─────────────────────────────────────────────
# FAISS Index Name
# ─────────────────────────────────────────────
FAISS_INDEX_NAME = "faiss_index"

# ─────────────────────────────────────────────
# Ensure directories exist
# ─────────────────────────────────────────────
os.makedirs(UPLOAD_DIR,      exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)