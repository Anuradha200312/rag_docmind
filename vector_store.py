import os
import uuid
import logging
import math
from functools import lru_cache
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    has_qdrant = True
except ImportError:
    has_qdrant = False
    QdrantClient = None
    qmodels = None

try:
    from sentence_transformers import SentenceTransformer
    has_sentence_transformers = True
except ImportError:
    has_sentence_transformers = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)
load_dotenv()

# Embedding config
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ─────────────────────────────────────────────────────────────────
# Embedding Generation (SentenceTransformer)
# ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load the SentenceTransformer model once."""
    if not has_sentence_transformers:
        raise ImportError("sentence-transformers package is required for Qdrant RAG pipeline. Please install it using: pip install sentence-transformers")
    logger.info("Loading SentenceTransformer model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_sentence_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for a list of text chunks."""
    model = _get_embedding_model()
    try:
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vec.tolist() for vec in vectors]
    except Exception as e:
        logger.error("Failed to generate sentence embeddings: %s", e)
        try:
            import streamlit as st
            st.warning(f"⚠️ Embedding generation error: {e}")
        except Exception:
            pass
        # Return zero vectors fallback
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]

def get_single_embedding(text: str) -> list[float]:
    """Generate vector embedding for a query string."""
    return get_sentence_embeddings([text])[0]

# ─────────────────────────────────────────────────────────────────
# Qdrant Vector Store
# ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Initialize QdrantClient based on environment variables for Qdrant Cloud only."""
    if not has_qdrant:
        raise ImportError("qdrant-client package is required for Qdrant RAG pipeline. Please install it using: pip install qdrant-client")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be defined in your environment variables for Qdrant Cloud.")

    logger.info("Connecting to Qdrant Cloud at %s", qdrant_url)
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False, timeout=60.0)

def get_qdrant_collection_name(user_id: str, chat_id: str, filename: str) -> str:
    """
    Namespace collection name safely for Qdrant.
    Includes user_id, chat_id, and the first 10 sanitized characters of the PDF name.
    """
    import re
    clean_uid = user_id.replace("-", "_")
    clean_cid = chat_id.replace("-", "_")
    
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Strip any characters not alphanumeric or underscore
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
    
    # Extract first 10 characters
    pdf_part = sanitized_name[:10]
    if not pdf_part:
        pdf_part = "doc"
        
    # Trim underscores and make lowercase
    pdf_part = pdf_part.lower().strip("_")
    if not pdf_part:
        pdf_part = "doc"
        
    return f"u_{clean_uid}_c_{clean_cid}_p_{pdf_part}"

def ensure_qdrant_collection(collection_name: str) -> None:
    """Create collection if it doesn't already exist."""
    client = _get_qdrant_client()
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE
                )
            )
            logger.info("Created Qdrant collection: %s", collection_name)
    except Exception as e:
        logger.error("Failed to ensure collection %s: %s", collection_name, e)
        raise e

def qdrant_upsert_chunks(
    chat_id: str,
    user_id: str,
    filename: str,
    chunks: list[dict],
    embeddings: list[list[float]]
) -> str:
    """Upsert text chunks and their embeddings into Qdrant Cloud."""
    client = _get_qdrant_client()
    collection_name = get_qdrant_collection_name(user_id, chat_id, filename)
    ensure_qdrant_collection(collection_name)

    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Safe 64-bit int ID generator
        point_id = abs(hash(f"{chat_id}_{idx}")) % (2**63)
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["chunk_text"],
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points, wait=True)
    logger.info("Upserted %d points to Qdrant collection %s", len(points), collection_name)
    return collection_name

def qdrant_search(collection_name: str, query_text: str, top_k: int = 5) -> list[dict]:
    """Search Qdrant Cloud for similar chunks based on a query string."""
    client = _get_qdrant_client()
    query_vector = get_single_embedding(query_text)

    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [
            {
                "chunk_text": r.payload.get("chunk_text", ""),
                "page_number": r.payload.get("page_number", 0),
                "chunk_index": r.payload.get("chunk_index", 0),
                "score": r.score
            }
            for r in results if r.payload
        ]
    except Exception as e:
        logger.error("Qdrant search error for collection %s: %s", collection_name, e)
        try:
            import streamlit as st
            st.warning(f"⚠️ Qdrant search error for collection {collection_name}: {e}")
        except Exception:
            pass
        return []

def qdrant_delete_collection(chat_id: str) -> None:
    """Delete a collection from Qdrant associated with a chat session from PostgreSQL lookup."""
    client = _get_qdrant_client()
    try:
        # Import dynamically to avoid circular import issues
        from database import SessionLocal, Document
        from sqlalchemy import select
        
        with SessionLocal() as session:
            stmt = select(Document).where(Document.chat_id == chat_id)
            doc = session.execute(stmt).scalar_one_or_none()
            if doc and doc.qdrant_collection:
                logger.info("Deleting Qdrant collection: %s", doc.qdrant_collection)
                client.delete_collection(collection_name=doc.qdrant_collection)
    except Exception as e:
        logger.info("Qdrant collection deletion skipped / failed: %s", e)

# ─────────────────────────────────────────────────────────────────
# ChromaDB (Local Ephemeral / Prototype Pipeline)
# ─────────────────────────────────────────────────────────────────
class ChromaStore:
    _store = {}  # chat_id (or col_name) -> list of chunks with embeddings

    def __init__(self):
        # Dummy client to support legacy check: st.session_state.chroma_store.client.get_collection(col_name).count()
        class DummyCollection:
            def __init__(self, store, col_name):
                self.store = store
                self.col_name = col_name
            def count(self):
                return len(self.store.get(self.col_name, []))

        class DummyClient:
            def __init__(self, store):
                self.store = store
            def get_collection(self, name):
                return DummyCollection(self.store, name)

        self.client = DummyClient(self._store)

    def get_collection_name(self, chat_id: str) -> str:
        return f"chroma_{chat_id.replace('-', '_')}"

    def _embed(self, text: str) -> list:
        text = text.lower()
        vec = [text.count(chr(i)) for i in range(97, 123)]
        total = sum(vec) or 1
        return [v / total for v in vec]

    def _cosine(self, a, b):
        dot = sum(x*y for x,y in zip(a,b))
        mag = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
        return dot / (mag + 1e-9)

    def index_chunks(self, chat_id: str, chunks: list[dict]) -> None:
        col_name = self.get_collection_name(chat_id)
        self._store[col_name] = [
            {**c, "vec": self._embed(c["chunk_text"])} for c in chunks
        ]

    def search(self, chat_id: str, query_text: str, top_k: int = 3) -> list[dict]:
        col_name = self.get_collection_name(chat_id)
        if col_name not in self._store:
            return []
        q_vec = self._embed(query_text)
        scored = sorted(
            self._store[col_name],
            key=lambda c: self._cosine(q_vec, c["vec"]),
            reverse=True
        )
        return [{"chunk_text": c["chunk_text"], "page_number": c["page_number"],
                 "chunk_index": c["chunk_index"], "distance": 0.1} for c in scored[:top_k]]
