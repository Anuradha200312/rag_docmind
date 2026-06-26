import os
import uuid
import datetime
import hashlib
from dotenv import load_dotenv
from sqlalchemy import (
    String, Text, Integer, DateTime, ForeignKey, select, delete, func, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Load environment variables
load_dotenv()

# Streamlit uses synchronous connection. Convert pg+asyncpg connection to standard pg+psycopg2
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:pyadmin@localhost:5432/docmind")
SYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

# Setup synchronous engine
engine = create_engine(
    SYNC_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

class Base(DeclarativeBase):
    pass

# Password hashing configuration (Zero-dependency PBKDF2)
def hash_password(plain_password: str) -> str:
    salt = os.urandom(16)
    db_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt, 100000)
    return f"{salt.hex()}:{db_hash.hex()}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        salt_hex, hash_hex = hashed_password.split(':')
        salt = bytes.fromhex(salt_hex)
        db_hash = bytes.fromhex(hash_hex)
        new_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt, 100000)
        return new_hash == db_hash
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users_hybrid"

    user_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chats: Mapped[list["Chat"]] = relationship(
        "Chat", back_populates="user", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )


class Chat(Base):
    __tablename__ = "chats_hybrid"

    chat_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users_hybrid.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="chats")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="chat", cascade="all, delete-orphan", order_by="Message.timestamp"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="chat", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages_hybrid"

    message_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    chat_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chats_hybrid.chat_id", ondelete="CASCADE"), nullable=False, index=True
    )
    sender: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' or 'assistant'
    message: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chat: Mapped[Chat] = relationship("Chat", back_populates="messages")


class Document(Base):
    __tablename__ = "documents_hybrid"

    document_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users_hybrid.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    chat_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("chats_hybrid.chat_id", ondelete="SET NULL"), nullable=True, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    pipeline_used: Mapped[str] = mapped_column(String(20), default="direct")  # 'direct' or 'rag'
    qdrant_collection: Mapped[str | None] = mapped_column(String(255), nullable=True)
    upload_time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="documents")
    chat: Mapped["Chat"] = relationship("Chat", back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks_hybrid"

    chunk_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents_hybrid.document_id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")

# ─────────────────────────────────────────────────────────────────
# Synchronous Helper Functions
# ─────────────────────────────────────────────────────────────────

def init_db():
    """Initializes database and tables synchronously, falling back to SQLite if PostgreSQL is unavailable."""
    global engine
    from urllib.parse import urlparse
    import psycopg2
    
    use_sqlite = False
    try:
        result = urlparse(SYNC_DATABASE_URL)
        dbname = result.path.lstrip('/')
        
        # Connect to postgres default DB to check target DB existence
        conn = psycopg2.connect(
            host=result.hostname or 'localhost',
            port=result.port or 5432,
            user=result.username or 'postgres',
            password=result.password or '',
            database='postgres',
            connect_timeout=3
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check database
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {dbname}")
            print(f"Created database: {dbname}")
        conn.close()
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}. Falling back to local SQLite database.")
        use_sqlite = True

    if use_sqlite or "sqlite" in SYNC_DATABASE_URL.lower():
        engine = create_engine(
            "sqlite:///docmind.db",
            connect_args={"check_same_thread": False}
        )
        SessionLocal.configure(bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

def db_register_user(name: str, email: str, password_plain: str) -> dict | None:
    """Registers a new user synchronously."""
    with SessionLocal() as session:
        try:
            # Check unique email
            stmt = select(User).where(User.email == email)
            existing = session.execute(stmt).scalar_one_or_none()
            if existing:
                return None
            
            user = User(
                name=name,
                email=email,
                password_hash=hash_password(password_plain)
            )
            session.add(user)
            session.commit()
            return {"user_id": user.user_id, "name": user.name, "email": user.email}
        except Exception as e:
            session.rollback()
            raise e

def db_verify_user(email: str, password_plain: str) -> dict | None:
    """Verifies user credentials synchronously."""
    with SessionLocal() as session:
        stmt = select(User).where(User.email == email)
        user = session.execute(stmt).scalar_one_or_none()
        if user and verify_password(password_plain, user.password_hash):
            return {"user_id": user.user_id, "name": user.name, "email": user.email}
        return None

def db_create_chat(user_id: str, title: str = "New Chat") -> dict:
    """Creates a new chat session synchronously."""
    with SessionLocal() as session:
        try:
            chat = Chat(user_id=user_id, title=title)
            session.add(chat)
            session.commit()
            return {"chat_id": chat.chat_id, "user_id": chat.user_id, "title": chat.title}
        except Exception as e:
            session.rollback()
            raise e

def db_get_user_chats(user_id: str) -> list[dict]:
    """Retrieves all chat sessions for a user synchronously."""
    with SessionLocal() as session:
        stmt = select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
        chats = session.execute(stmt).scalars().all()
        return [{"chat_id": c.chat_id, "user_id": c.user_id, "title": c.title, "updated_at": c.updated_at} for c in chats]

def db_delete_chat(chat_id: str) -> bool:
    """Deletes a chat session synchronously."""
    with SessionLocal() as session:
        try:
            stmt = delete(Chat).where(Chat.chat_id == chat_id)
            session.execute(stmt)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e

def db_load_chat_history(chat_id: str) -> list[dict]:
    """Loads all messages for a chat session synchronously."""
    with SessionLocal() as session:
        stmt = select(Message).where(Message.chat_id == chat_id).order_by(Message.timestamp.asc())
        messages = session.execute(stmt).scalars().all()
        return [
            {
                "message_id": m.message_id,
                "role": m.sender,
                "content": m.message,
                "timestamp": m.timestamp.strftime("%I:%M %p") if m.timestamp else ""
            }
            for m in messages
        ]

def db_save_message(chat_id: str, role: str, content: str) -> dict:
    """Saves a message synchronously and updates chat session's updated_at field."""
    with SessionLocal() as session:
        try:
            msg = Message(chat_id=chat_id, sender=role, message=content)
            session.add(msg)
            
            stmt = select(Chat).where(Chat.chat_id == chat_id)
            chat = session.execute(stmt).scalar_one_or_none()
            if chat:
                chat.updated_at = datetime.datetime.utcnow()
                
            session.commit()
            return {
                "message_id": msg.message_id,
                "role": msg.sender,
                "content": msg.message,
                "timestamp": msg.timestamp.strftime("%I:%M %p") if msg.timestamp else ""
            }
        except Exception as e:
            session.rollback()
            raise e

def db_save_document(
    user_id: str, chat_id: str, filename: str, token_count: int, pipeline_used: str, qdrant_collection: str | None, chunks: list[dict]
) -> dict:
    """Saves metadata for an uploaded document and its chunks synchronously."""
    with SessionLocal() as session:
        try:
            stmt = select(Document).where(Document.chat_id == chat_id)
            existing_doc = session.execute(stmt).scalar_one_or_none()
            if existing_doc:
                session.delete(existing_doc)

            doc = Document(
                user_id=user_id,
                chat_id=chat_id,
                filename=filename,
                token_count=token_count,
                pipeline_used=pipeline_used,
                qdrant_collection=qdrant_collection
            )
            session.add(doc)
            session.flush() # Get document_id
            
            db_chunks = [
                DocumentChunk(
                    document_id=doc.document_id,
                    page_number=c["page_number"],
                    chunk_index=c["chunk_index"],
                    chunk_text=c["chunk_text"]
                )
                for c in chunks
            ]
            session.add_all(db_chunks)
            
            session.commit()
            return {
                "document_id": doc.document_id,
                "chat_id": doc.chat_id,
                "filename": doc.filename,
                "token_count": doc.token_count,
                "pipeline_used": doc.pipeline_used,
                "qdrant_collection": doc.qdrant_collection
            }
        except Exception as e:
            session.rollback()
            raise e

def db_get_document_chunks(document_id: str) -> list[dict]:
    """Retrieves all chunks for a document synchronously."""
    with SessionLocal() as session:
        stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index.asc())
        chunks = session.execute(stmt).scalars().all()
        return [{"page_number": c.page_number, "chunk_index": c.chunk_index, "chunk_text": c.chunk_text} for c in chunks]

def db_get_chat_document(chat_id: str) -> dict | None:
    """Retrieves document associated with the chat session synchronously."""
    with SessionLocal() as session:
        stmt = select(Document).where(Document.chat_id == chat_id)
        doc = session.execute(stmt).scalar_one_or_none()
        if doc:
            return {
                "document_id": doc.document_id,
                "chat_id": doc.chat_id,
                "filename": doc.filename,
                "token_count": doc.token_count,
                "pipeline_used": doc.pipeline_used,
                "qdrant_collection": doc.qdrant_collection
            }
        return None
