"""
Development database seeder.
Run ONLY in development environment to populate test.db with mock data.
Usage from project root:
    python -m app.backend.functionality.seed_dev_db
Usage from app/backend directory:
    python functionality/seed_dev_db.py
"""
import sys
import os
from pathlib import Path

# ==============================
# PATH SETUP (works from any cwd)
# ==============================
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent  # app/backend

# Add backend directory to path for consistent imports
sys.path.insert(0, str(backend_dir))

# Use RELATIVE imports to match api.py (prevents duplicate table definitions)
from database import (
    engine,
    create_db_and_tables,
    get_direct_session,
    ENVIRONMENT
)
from models import (
    Section, Student, TestInstance, TestItem,
    TestPaperInstance, StudentAnswer
)

from sqlmodel import delete
import warnings

# ==============================
# MOCK DATA
# ==============================
MOCK_DATA = {
        "sections": [
            {"section_id": 1, "section": "3-Rizal"},
            {"section_id": 2, "section": "3-Aguinaldo"}
        ],
        "students": [
            {"student_no": "202160151", "name": "Mohammad Hamdi Tuan", "section_id": 1},
            {"student_no": "202011111", "name": "Jose Ernesto Tomanan", "section_id": 1},
            {"student_no": "202022222", "name": "Gabriel Abilla", "section_id": 2},
            {"student_no": "202033333", "name": "David Salon", "section_id": 2}
        ],
        "test_instances": [
            {
                "test_id": "3-Rizal_Seatwork-1",
                "name": "Seatwork-1",
                "section_id": 1,
                "date": "2025-11-11",
                "is_done_rendering": False
            },
            {
                "test_id": "3-Aguinaldo_Quiz-1",
                "name": "Quiz-1",
                "section_id": 2,
                "date": "2026-01-12",
                "is_done_rendering": False
            }
        ],
        "test_items": [
            {
                "item_id": 1,
                "test_id": "3-Rizal_Seatwork-1",
                "label": "Problem 1",
                "question": "Solve for x: 2x + 5 = 15",
                "is_problem_solving": True,
                "expected_answer_rubric_questions": "Correct equation setup (2pts); Accurate solution (2pts);"
            },
            {
                "item_id": 2,
                "test_id": "3-Rizal_Seatwork-1",
                "label": "Question 2",
                "question": "What is the capital of France?",
                "is_problem_solving": False,
                "expected_answer_rubric_questions": "Paris (1pt)"
            },
            {
                "item_id": 3,
                "test_id": "3-Aguinaldo_Quiz-1",
                "label": "Problem 1",
                "question": "Calculate the area of a circle with radius 7 cm",
                "is_problem_solving": True,
                "expected_answer_rubric_questions": "Correct formula (2pts); Correct substitution (1pt);"
            }
        ]
        }



# ==============================
# SEEDING LOGIC
# ==============================
def seed_dev_database():
    """Reset database and insert mock development data"""
    # SAFETY CHECK: Block production seeding
    if ENVIRONMENT != "development":
        raise RuntimeError(
            f"ABORTED: Seeding only allowed in 'development' environment. "
            f"Current ENVIRONMENT={ENVIRONMENT}. "
            f"Check your .env file!"
        )
    
    # Verify SQLite dev database
    db_url = str(engine.url)
    if "sqlite" not in db_url.lower() or "test.db" not in db_url:
        warnings.warn(
            f"Warning: Seeding non-standard database: {db_url}\n"
            f"Ensure this is intentional for development!",
            UserWarning
        )
    
    print(f"Seeding development database (ENVIRONMENT={ENVIRONMENT})...")
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
        
        # INSERT mock data
        for data in MOCK_DATA["sections"]:
            session.add(Section(**data))
        for data in MOCK_DATA["students"]:
            session.add(Student(**data))
        for data in MOCK_DATA["test_instances"]:
            session.add(TestInstance(**data))
        for data in MOCK_DATA["test_items"]:
            session.add(TestItem(**data))
        session.commit()
        
        print("SUCCESS: Database seeded with mock development data")
        print(f"Sections: {len(MOCK_DATA['sections'])} | Students: {len(MOCK_DATA['students'])} | Tests: {len(MOCK_DATA['test_instances'])}")
        return True
        
    except Exception as e:
        session.rollback()
        print(f"FAILED: Seeding error - {type(e).__name__}: {e}")
        raise
    finally:
        session.close()



# ==============================
# CLI EXECUTION
# ==============================
if __name__ == "__main__":
    try:
        seed_dev_database()
        sys.exit(0)
    except Exception as e:
        print(f"\nSeeding aborted. Fix errors and retry.", file=sys.stderr)
        sys.exit(1)