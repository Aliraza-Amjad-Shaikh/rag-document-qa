import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────
# Folder Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ─────────────────────────────────────────────
# Model Names
# ─────────────────────────────────────────────
VISION_MODEL = "gpt-4o-mini"  # Used for image description (vision-capable)
CHAT_MODEL = "gpt-4o-mini"  # Used for answer generation
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embeddings (1536-dim)

# ─────────────────────────────────────────────
# Chunking Settings
# ─────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ─────────────────────────────────────────────
# Semantic Chunking Settings
# ─────────────────────────────────────────────
SEMANTIC_BREAKPOINT_TYPE = "percentile"  # Options: percentile, standard_deviation, interquartile
SEMANTIC_BREAKPOINT_THRESHOLD = 85  # Higher = fewer, larger chunks. Lower = more, smaller chunks
MAX_SEMANTIC_CHUNK_SIZE = 1500  # Hard cap — semantic chunks larger than this get split further
MIN_SEMANTIC_CHUNK_SIZE = 50  # Hard floor — chunks smaller than this get dropped

# ─────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────
TOP_K_RESULTS = 6

# ─────────────────────────────────────────────
# Confidence Score Thresholds
# ─────────────────────────────────────────────
# NOTE: These were calibrated for MiniLM (384-dim) similarity distribution.
# Must be re-tuned after switching to OpenAI text-embedding-3-small (1536-dim).
# Rebuild the FAISS index, run sample queries, log normalized scores, then adjust.
CONFIDENCE_HIGH = 0.45
CONFIDENCE_MEDIUM = 0.55

# ─────────────────────────────────────────────
# Supported File Types
# ─────────────────────────────────────────────
PDF_EXTENSIONS = [".pdf"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS + IMAGE_EXTENSIONS

# ─────────────────────────────────────────────
# Query Rewriting (HyDE-lite)
# ─────────────────────────────────────────────
ENABLE_QUERY_REWRITING = False  # Toggle to disable without code changes
QUERY_REWRITE_TEMPERATURE = 0  # Deterministic rewrites

# ─────────────────────────────────────────────
# FAISS Index Name
# ─────────────────────────────────────────────
FAISS_INDEX_NAME = "faiss_index"

# ─────────────────────────────────────────────
# Ensure directories exist
# ─────────────────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)