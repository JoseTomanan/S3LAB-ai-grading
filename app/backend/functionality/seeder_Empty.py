"""
Development database seeder for an EMPTY dev DB.
Run ONLY in development environment to create/reset an empty test.db (no rows).
Usage from project root:
    python -m app.backend.functionality.seeder_Empty
Usage from app/backend directory:
    python functionality/seeder_Empty.py
"""
import sys
from pathlib import Path



# ==============================
# region PATH SETUP (works from any cwd)
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent  # app/backend

# Add backend directory to path for consistent imports
sys.path.insert(0, str(backend_dir))

# Use RELATIVE imports to match api.py (prevents duplicate table definitions)
from database import (
    engine,
    create_db_and_tables,
    get_direct_session,
    ENVIRONMENT,
)
from models import (
    Section,
    Student,
    TestInstance,
    TestItem,
    TestPaperInstance,
    StudentAnswer,
)

from sqlmodel import delete
import warnings
# endregion
# ==============================


# ==============================
# region EMPTY SEEDING LOGIC
def seed_empty_database():
    """Reset database schema and leave all tables empty (no data)."""
    # SAFETY CHECK: Block production seeding
    if ENVIRONMENT != "development":
        raise RuntimeError(
            f"ABORTED: Empty seeding only allowed in 'development' environment. "
            f"Current ENVIRONMENT={ENVIRONMENT}. "
            f"Check your .env file!"
        )

    # Verify SQLite dev database
    db_url = str(engine.url)
    if "sqlite" not in db_url.lower() or "test.db" not in db_url:
        warnings.warn(
            f"Warning: Empty-seeding non-standard database: {db_url}\n"
            f"Ensure this is intentional for development!",
            UserWarning,
        )

    print(f"Resetting development database to EMPTY state (ENVIRONMENT={ENVIRONMENT})...")
    create_db_and_tables()  # Ensure tables exist

    session = get_direct_session()
    try:
        # DELETE in FK-safe order (reverse dependency chain)
        session.exec(delete(StudentAnswer))
        session.exec(delete(TestItem))
        session.exec(delete(TestPaperInstance))
        session.exec(delete(TestInstance))
        session.exec(delete(Student))
        session.exec(delete(Section))
        session.commit()

        print("SUCCESS: Database reset to empty state (no rows).")
        return True

    except Exception as e:
        session.rollback()
        print(f"FAILED: Empty seeding error - {type(e).__name__}: {e}")
        raise
    finally:
        session.close()


# endregion
# ==============================


if __name__ == "__main__":
    try:
        seed_empty_database()
        sys.exit(0)
    except Exception:
        print("\nEmpty seeding aborted. Fix errors and retry.", file=sys.stderr)
        sys.exit(1)