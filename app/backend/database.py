# database.py
from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
from fastapi import Depends

# Use SQLite for development
DATABASE_URL = "sqlite:///./test.db"

# Create engine with SQLite-specific args
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=False  # Set to True for SQL query logging
)

def create_db_and_tables():
    """Create all tables defined in SQLModel models"""
    SQLModel.metadata.create_all(engine)

# FastAPI dependency for session injection (recommended for endpoints)
def get_session() -> Generator[Session, None, None]:
    """Yield a database session (auto-cleanup)"""
    with Session(engine) as session:
        yield session

# Direct session for manual use (with explicit close)
def get_direct_session() -> Session:
    """Return a session for direct use (caller must close it)"""
    return Session(engine)