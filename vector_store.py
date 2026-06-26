import os
import uuid
import logging
from functools import lru_cache
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

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
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)

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
    def __init__(self):
        # Ephemeral client is lightweight and memory-only
        self.client = chromadb.EphemeralClient()

    def get_collection_name(self, chat_id: str) -> str:
        # Chroma collection names must match a regex pattern
        return f"chroma_{chat_id.replace('-', '_')}"

    def index_chunks(self, chat_id: str, chunks: list[dict]) -> None:
        """Store document chunks in ChromaDB local ephemeral client."""
        col_name = self.get_collection_name(chat_id)
        # Delete if exists
        try:
            self.client.delete_collection(col_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=col_name,
            embedding_function=DefaultEmbeddingFunction()
        )

        documents = [c["chunk_text"] for c in chunks]
        metadatas = [{"page_number": c["page_number"], "chunk_index": c["chunk_index"]} for c in chunks]
        ids = [f"{chat_id}_chunk_{c['chunk_index']}" for c in chunks]

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("ChromaDB: Indexed %d chunks in collection %s", len(chunks), col_name)

    def search(self, chat_id: str, query_text: str, top_k: int = 3) -> list[dict]:
        """Search ChromaDB collection for relevant chunks."""
        col_name = self.get_collection_name(chat_id)
        try:
            collection = self.client.get_collection(
                name=col_name,
                embedding_function=DefaultEmbeddingFunction()
            )
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            return [
                {
                    "chunk_text": doc,
                    "page_number": meta.get("page_number", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist
                }
                for doc, meta, dist in zip(docs, metas, distances)
            ]
        except Exception as e:
            logger.error("ChromaDB search error for chat %s: %s", chat_id, e)
            return []
