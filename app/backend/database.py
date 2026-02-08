from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
from fastapi import Depends
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================
# Database Configuration
# ==============================

# Environment-based database URL
# Use SQLite for development, PostgreSQL/MySQL for production
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    # PostgreSQL example (adjust for production DB)
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/assessment_db"
    )
    # Production engine settings
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
else:
    # SQLite for development/testing
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    # SQLite-specific engine settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )


# ==============================
# Database Initialization
# ==============================

def create_db_and_tables():
    """
    Create all tables defined in SQLModel models.
    Safe to call multiple times (idempotent).
    """
    SQLModel.metadata.create_all(engine)


def drop_all_tables():
    """
    Drop all tables (use with caution - for testing/reset only).
    """
    SQLModel.metadata.drop_all(engine)


# ==============================
# Session Management
# ==============================

def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for session injection.
    Yields a database session with automatic cleanup (rollback on exception, commit on success).
    
    Usage in endpoints:
        @app.post("/items")
        def create_item(session: Session = Depends(get_session)):
            ...
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Alternative: Direct session for manual use (caller must manage lifecycle)
def get_direct_session() -> Session:
    """
    Return a session for direct use (caller must close it).
    Use this for scripts, background tasks, or manual operations.
    
    Example:
        session = get_direct_session()
        try:
            # ... operations
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
    """
    return Session(engine)


# ==============================
# Helper Functions
# ==============================

def get_engine():
    """
    Return the database engine (for advanced operations like raw SQL).
    """
    return engine


def reset_database():
    """
    Drop and recreate all tables (development/testing only).
    WARNING: This will delete all data!
    """
    drop_all_tables()
    create_db_and_tables()