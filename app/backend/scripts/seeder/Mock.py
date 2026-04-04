"""
Development database seeder.
Run ONLY in development environment to populate test.db with mock data.
"""
import sys
from pathlib import Path



# ==============================
#region PATH SETUP (works from any cwd)
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent  # app/backend

# Add backend directory to path for consistent imports
sys.path.insert(0, str(backend_dir))

# Use RELATIVE imports to match api.py (prevents duplicate table definitions)
from core.database import (
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
#endregion
# ==============================



# ==============================
#region MOCK DATA
MOCK_DATA = {
        "sections": [
            {"section_id": 1, "section": "3-Rizal"},
            {"section_id": 2, "section": "3-Aguinaldo"}
        ],
        "students": [
            ## Section 1 students
            {"student_no": "200033333", "name": "Thomas Yorke", "section_id": 1},
            {"student_no": "200044444", "name": "Liam Gallagher", "section_id": 1},
            {"student_no": "200055555", "name": "Pierre Bouvier", "section_id": 1},
            {"student_no": "200066666", "name": "Chester Bennington", "section_id": 1},
            {"student_no": "200022222", "name": "John Rzeznik", "section_id": 1},
            {"student_no": "200011111", "name": "Billy Joe Armstrong", "section_id": 1},
            ## Section 2 students
            {"student_no": "202090011", "name": "David Amdahl", "section_id": 2},
            {"student_no": "202090022", "name": "Olivia Rodrigo Duterte", "section_id": 2},
            {"student_no": "202090033", "name": "Julius Babao", "section_id": 2},
            {"student_no": "202090044", "name": "Paulo Costa", "section_id": 2},
            {"student_no": "202090055", "name": "Nikola Kojic Soap", "section_id": 2},
        ],
        "test_instances": [
            {
                "test_id": "3-Rizal_Seatwork-1",
                "name": "Seatwork-1",
                "section_id": 1,
                "date": "2025-11-11T14:30:00Z",
                "is_done_rendering": False
            },
            {
                "test_id": "3-Aguinaldo_Quiz-1",
                "name": "Quiz-1",
                "section_id": 2,
                "date": "2026-01-12T14:30:00Z",
                "is_done_rendering": False
            }
        ],
        "test_items": [
            {
                "item_id": 1,
                "test_id": "3-Rizal_Seatwork-1",
                "label": "1",
                "question": "Solve for x: 2x + 5 = 15",
                "is_problem_solving": True,
                "expected_answer_rubric_questions": "Correct equation setup [2pts]; Accurate solution [2pts]"
            },
            {
                "item_id": 2,
                "test_id": "3-Rizal_Seatwork-1",
                "label": "2",
                "question": "What is the capital of France?",
                "is_problem_solving": False,
                "expected_answer_rubric_questions": "Paris [1pt]"
            },
            {
                "item_id": 5,
                "test_id": "3-Rizal_Seatwork-1",
                "label": "2b",
                "question": "What is the full name of the Filipino national hero?",
                "is_problem_solving": False,
                "expected_answer_rubric_questions": "Jose Protacio Rizal Mercado y Alonso Realonda [1pt]"
            },
            {
                "item_id": 3,
                "test_id": "3-Aguinaldo_Quiz-1",
                "label": "1a",
                "question": "What is the place value of the digit 7 in the number 7,249?",
                "is_problem_solving": False,
                "expected_answer_rubric_questions": "Thousands [1pt]"
            },
            {
                "item_id": 4,
                "test_id": "3-Aguinaldo_Quiz-1",
                "label": "3",
                "question": "A school library has 9,000 books. If 5,672 books are English books and the rest are Filipino books, how many Filipino books are there?",
                "is_problem_solving": True,
                "expected_answer_rubric_questions": "Correct operation/equation [2pts]; Correct calculation/solution [2pts]; Correct label in final answer (Filipino books) [1pts]"
            },
            {
                "item_id": 6,
                "test_id": "3-Aguinaldo_Quiz-1",
                "label": "2",
                "question": "Mang Juan harvested 2,450 mangoes on Monday and 3,125 mangoes on Tuesday. How many mangoes did he harvest in total?",
                "is_problem_solving": True,
                "expected_answer_rubric_questions": "Correct operation/equation [2pts]; Correct calculation/solution [2pts]; Correct label in final answer (mangoes) [1pts]"
            },
            {
                "item_id": 7,
                "test_id": "3-Aguinaldo_Quiz-1",
                "label": "1b",
                "question": "Write the number eight thousand, fifty-six in standard symbols.",
                "is_problem_solving": False,
                "expected_answer_rubric_questions": "8,056 [1pt]"
            },
        ]
    }
#endregion
# ==============================



# ==============================
#region SEEDING LOGIC
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
#endregion
# ==============================



if __name__ == "__main__":
    try:
        seed_dev_database()
        sys.exit(0)
    except Exception as e:
        print(f"\nSeeding aborted. Fix errors and retry.", file=sys.stderr)
        sys.exit(1)