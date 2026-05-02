import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables from .env file
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# OpenAI API Key
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────
# Folder Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR      = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ─────────────────────────────────────────────
# Model Names
# ─────────────────────────────────────────────
VISION_MODEL    = "gpt-4o"          # Used for image description (Phase 2)
EMBEDDING_MODEL = "text-embedding-3-small"  # Used for FAISS embeddings (Phase 3)
CHAT_MODEL      = "gpt-4o"          # Used for answer generation (Phase 5)

# ─────────────────────────────────────────────
# Chunking Settings
# ─────────────────────────────────────────────
CHUNK_SIZE      = 1000   # Max characters per chunk
CHUNK_OVERLAP   = 150    # Overlap between chunks to preserve context

# ─────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────
TOP_K_RESULTS   = 6     # Number of chunks to retrieve per query

# ─────────────────────────────────────────────
# Confidence Score Thresholds
# ─────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.60   # Score above this → High confidence
CONFIDENCE_MEDIUM = 0.40   # Score above this → Medium confidence
                            # Score below this → Low confidence → "I don't know"

# ─────────────────────────────────────────────
# Supported File Types
# ─────────────────────────────────────────────
PDF_EXTENSIONS   = [".pdf"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS + IMAGE_EXTENSIONS

# ─────────────────────────────────────────────
# FAISS Vector Store File Names
# ─────────────────────────────────────────────
FAISS_INDEX_NAME = "faiss_index"   # FAISS will save as faiss_index.faiss + faiss_index.pkl

# ─────────────────────────────────────────────
# Ensure required directories exist at startup
# ─────────────────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)