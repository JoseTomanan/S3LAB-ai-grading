import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select, delete
from datetime import datetime
import io
import json
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from core.database import engine, create_db_and_tables
from models import *

# Test client
client = TestClient(app)



# ==============================
#region Mock Data
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
                "expected_answer_rubric_questions": "Correct equation setup (2pts); Accurate solution (2pts)"
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
                "expected_answer_rubric_questions": "Correct formula (2pts); Correct substitution (1pt)"
            }
            ]
        }
#endregion


# ==============================
#region Test Fixtures
# ==============================
@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Setup and teardown database for each test"""
    # Create tables
    create_db_and_tables()

    # Populate with mock data
    with Session(engine) as session:
        # Clear existing data
        session.exec(delete(StudentAnswer)) # type: ignore
        session.exec(delete(TestItem)) # type: ignore
        session.exec(delete(TestPaperInstance)) # type: ignore
        session.exec(delete(TestInstance)) # type: ignore
        session.exec(delete(Student)) # type: ignore
        session.exec(delete(Section)) # type: ignore
        session.commit()

        # Add sections
        for sec_data in MOCK_DATA["sections"]:
            session.add(Section(**sec_data))

        # Add students
        for stud_data in MOCK_DATA["students"]:
            session.add(Student(**stud_data))

        # Add test instances
        for inst_data in MOCK_DATA["test_instances"]:
            session.add(TestInstance(**inst_data))

        # Add test items
        for item_data in MOCK_DATA["test_items"]:
            session.add(TestItem(**item_data))

        session.commit()

    # Run test
    yield
#endregion


# ==============================
#region Helper Functions
# ==============================
def create_test_image():
    """Create a simple test image (white square)"""
    import numpy as np
    import cv2

    # Create white image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return io.BytesIO(buffer.tobytes())
#endregion


# ==============================
#region Health Check
# ==============================
def test_health_check():
    """Test GET /health"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
#endregion


# ==============================
#region Section Endpoints Tests
# ==============================
def test_get_all_sections_success():
    """Test GET /api/sections - Success"""
    response = client.get("/api/sections")

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["section_id"] == 1
    assert data[0]["section_name"] == "3-Rizal"
    assert data[1]["section_id"] == 2
    assert data[1]["section_name"] == "3-Aguinaldo"


def test_get_students_in_section_success():
    """Test GET /api/sections/{section_id} - Success"""
    response = client.get("/api/sections/1")

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["student_no"] == "202160151"
    assert data[0]["name"] == "Mohammad Hamdi Tuan"
    assert data[0]["section_id"] == 1


def test_get_students_in_section_not_found():
    """Test GET /api/sections/{section_id} - Section not found"""
    response = client.get("/api/sections/999")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_add_new_section_success():
    """Test POST /api/sections - Success"""
    response = client.post("/api/sections", json={"section": "3-Bonifacio"})

    assert response.status_code == 201
    data = response.json()
    assert data["section_name"] == "3-Bonifacio"
    assert "section_id" in data


def test_add_new_section_duplicate():
    """Test POST /api/sections - Duplicate section name"""
    response = client.post("/api/sections", json={"section": "3-Rizal"})

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "already exists" in data["detail"].lower()
#endregion


# ==============================
#region Student Endpoints Tests
# ==============================
def test_add_new_student_success():
    """Test POST /api/students - Success"""
    new_student = {
        "name": "New Student",
        "student_no": "202199999",
        "section_id": 1
    }

    response = client.post("/api/students", json=new_student)

    assert response.status_code == 201
    data = response.json()

    assert data["student_no"] == "202199999"
    assert data["name"] == "New Student"
    assert data["section_id"] == 1


def test_add_new_student_section_not_found():
    """Test POST /api/students - Section does not exist"""
    invalid_student = {
        "name": "Invalid Section Student",
        "student_no": "202188888",
        "section_id": 999  # Non-existent section
    }

    response = client.post("/api/students", json=invalid_student)

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "section" in data["detail"].lower()


def test_add_new_student_duplicate():
    """Test POST /api/students - Duplicate student number"""
    duplicate_student = {
        "name": "Duplicate",
        "student_no": "202160151",  # Already exists
        "section_id": 2
    }

    response = client.post("/api/students", json=duplicate_student)

    assert response.status_code == 409
    data = response.json()
    assert "detail" in data
    assert "already exists" in data["detail"].lower()


def test_edit_student_details_success():
    """Test PATCH /api/students/{student_no} - Success"""
    update_data = {
        "name": "Updated Name",
        "section_id": 2
    }

    response = client.patch("/api/students/202160151", json=update_data)

    assert response.status_code == 200
    data = response.json()

    assert data["student_no"] == "202160151"
    assert data["name"] == "Updated Name"
    assert data["section_id"] == 2


def test_edit_student_details_not_found():
    """Test PATCH /api/students/{student_no} - Student not found"""
    update_data = {
        "name": "Non-existent",
        "section_id": 1
    }

    response = client.patch("/api/students/999999999", json=update_data)

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "student" in data["detail"].lower()


def test_delete_student_success():
    """Test DELETE /api/students/{student_no} - Success"""
    response = client.delete("/api/students/202160151")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_student_not_found():
    """Test DELETE /api/students/{student_no} - Student not found"""
    response = client.delete("/api/students/999999999")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "student" in data["detail"].lower()
#endregion


# ==============================
#region Test Instance Endpoints Tests
# ==============================
def test_get_test_instances_success():
    """Test GET /api/test_instances - Success"""
    response = client.get("/api/test_instances")

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["test_id"] == "3-Rizal_Seatwork-1"
    assert data[0]["name"] == "Seatwork-1"
    assert data[0]["section_id"] == 1


def test_get_test_instance_success():
    """Test GET /api/test_instances/{test_id} - Success"""
    response = client.get("/api/test_instances/3-Rizal_Seatwork-1")

    assert response.status_code == 200
    data = response.json()

    assert data["test_id"] == "3-Rizal_Seatwork-1"
    assert data["name"] == "Seatwork-1"
    assert data["section_id"] == 1
    assert len(data["items"]) == 2


def test_get_test_instance_not_found():
    """Test GET /api/test_instances/{test_id} - Not found"""
    response = client.get("/api/test_instances/non-existent-test")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_add_test_instance_success():
    """Test POST /api/test_instances - Success"""
    new_instance = {
        "name": "Quiz-2",
        "section_id": 1,
        "date": "2026-02-01"
    }

    response = client.post("/api/test_instances", json=new_instance)

    assert response.status_code == 201
    data = response.json()

    assert data["name"] == "Quiz-2"
    assert data["section_id"] == 1
    assert data["date"] == "2026-02-01"
    assert "test_id" in data
    assert data["is_done_rendering"] == False
    assert data["items"] == []


def test_add_test_instance_section_not_found():
    """Test POST /api/test_instances - Section not found"""
    invalid_instance = {
        "name": "Invalid Quiz",
        "section_id": 999,  # Non-existent section
        "date": "2026-02-01"
    }

    response = client.post("/api/test_instances", json=invalid_instance)

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "section" in data["detail"].lower()


def test_edit_test_instance_success():
    """Test PATCH /api/test_instances/{test_id} - Success"""
    update_data = {
        "date": "2026-01-15",
        "items": [
            {
                "item_id": 1,
                "label": "New Problem 1",
                "question": "New Question 1",
                "is_problem_solving": True,
                "expected_answer_rubric_questions": "New rubric"
            }
        ]
    }

    response = client.patch(
        "/api/test_instances/3-Rizal_Seatwork-1",
        json=update_data
    )

    assert response.status_code == 200
    data = response.json()

    assert data["test_id"] == "3-Rizal_Seatwork-1"
    assert data["date"] == "2026-01-15"
    assert len(data["items"]) == 1
    assert data["items"][0]["question"] == "New Question 1"


def test_edit_test_instance_not_found():
    """Test PATCH /api/test_instances/{test_id} - Not found"""
    update_data = {
        "date": "2026-01-15"
    }

    response = client.patch(
        "/api/test_instances/non-existent",
        json=update_data
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_delete_test_instance_success():
    """Test DELETE /api/test_instances/{test_id} - Success"""
    response = client.delete("/api/test_instances/3-Rizal_Seatwork-1")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_test_instance_not_found():
    """Test DELETE /api/test_instances/{test_id} - Not found"""
    response = client.delete("/api/test_instances/non-existent")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_export_test_results_success():
    """Test GET /api/test_instances/{test_id}/export - Success"""
    response = client.get("/api/test_instances/3-Rizal_Seatwork-1/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert "attachment" in response.headers["content-disposition"]
    assert "3-Rizal_Seatwork-1.xlsx" in response.headers["content-disposition"]


def test_export_test_results_not_found():
    """Test GET /api/test_instances/{test_id}/export - Not found"""
    response = client.get("/api/test_instances/non-existent/export")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()
#endregion


# ==============================
#region Test Item Endpoints Tests  (prefix: /api/test_items)
# ==============================
def test_get_test_instance_items_success():
    """Test GET /api/test_items/{test_id}/items - Success"""
    response = client.get("/api/test_items/3-Rizal_Seatwork-1/items")

    assert response.status_code == 200
    data = response.json()

    assert data["test_id"] == "3-Rizal_Seatwork-1"
    assert len(data["items"]) == 2
    assert data["items"][0]["item_id"] == 1
    assert data["items"][0]["label"] == "Problem 1"
    assert data["items"][0]["question"] == "Solve for x: 2x + 5 = 15"


def test_get_test_instance_items_not_found():
    """Test GET /api/test_items/{test_id}/items - Test not found"""
    response = client.get("/api/test_items/non-existent/items")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_add_test_item_success():
    """Test POST /api/test_items/{test_id}/items - Success"""
    new_item = {
        "label": "Problem 4",
        "question": "What is 2+2?",
        "is_problem_solving": False,
        "expected_answer_rubric_questions": "Correct answer: 4 (1pt)"
    }

    response = client.post(
        "/api/test_items/3-Rizal_Seatwork-1/items",
        json=new_item
    )

    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["label"] == "Problem 4"
    assert data["items"][0]["question"] == "What is 2+2?"
    assert data["items"][0]["item_id"] == 4  # max(1,2,3) + 1


def test_add_test_item_test_not_found():
    """Test POST /api/test_items/{test_id}/items - Test not found"""
    new_item = {
        "label": "Problem X",
        "question": "What is 2+2?",
        "is_problem_solving": False,
        "expected_answer_rubric_questions": "Correct answer: 4 (1pt)"
    }

    response = client.post(
        "/api/test_items/non-existent/items",
        json=new_item
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_edit_test_item_success():
    """Test PATCH /api/test_items/{test_id}/items/{item_id} - Success"""
    update_data = {
        "label": "Updated Problem 1",
        "question": "Solve for x: 3x + 6 = 21",
        "is_problem_solving": True,
        "expected_answer_rubric_questions": "Updated rubric"
    }

    response = client.patch(
        "/api/test_items/3-Rizal_Seatwork-1/items/1",
        json=update_data
    )

    assert response.status_code == 200
    data = response.json()

    assert data["item_id"] == 1
    assert data["label"] == "Updated Problem 1"
    assert data["question"] == "Solve for x: 3x + 6 = 21"


def test_edit_test_item_not_found():
    """Test PATCH /api/test_items/{test_id}/items/{item_id} - Item not found"""
    update_data = {
        "question": "Updated question"
    }

    response = client.patch(
        "/api/test_items/3-Rizal_Seatwork-1/items/999",
        json=update_data
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test item" in data["detail"].lower()


def test_delete_test_item_success():
    """Test DELETE /api/test_items/{test_id}/items/{item_id} - Success"""
    response = client.delete("/api/test_items/3-Rizal_Seatwork-1/items/1")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_test_item_not_found():
    """Test DELETE /api/test_items/{test_id}/items/{item_id} - Item not found"""
    response = client.delete("/api/test_items/3-Rizal_Seatwork-1/items/999")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test item" in data["detail"].lower()
#endregion


# ==============================
#region Student Answer Processing Tests  (prefix: /api/student_answers)
# ==============================
def test_process_student_answer_image_invalid_file():
    """Test POST /api/student_answers/{test_id}/{student_no}/image_preprocess - Invalid file type"""
    text_file = io.BytesIO(b"This is not an image")

    response = client.post(
        "/api/student_answers/3-Rizal_Seatwork-1/202160151/image_preprocess",
        files={"files": ("test.txt", text_file, "text/plain")}
    )

    assert response.status_code == 415
    data = response.json()
    assert "detail" in data
    assert "unsupported" in data["detail"].lower()


def test_process_student_answer_image_test_not_found():
    """Test POST /api/student_answers/{test_id}/{student_no}/image_preprocess - Test not found"""
    test_image = create_test_image()

    response = client.post(
        "/api/student_answers/non-existent/202160151/image_preprocess",
        files={"files": ("test.jpg", test_image, "image/jpeg")}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_process_student_answer_image_student_not_found():
    """Test POST /api/student_answers/{test_id}/{student_no}/image_preprocess - Student not found"""
    test_image = create_test_image()

    response = client.post(
        "/api/student_answers/3-Rizal_Seatwork-1/999999999/image_preprocess",
        files={"files": ("test.jpg", test_image, "image/jpeg")}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "student" in data["detail"].lower()


def test_label_save_boxes_invalid_file():
    """Test POST /api/student_answers/{test_id}/{student_no}/label_save_boxes - Invalid file type"""
    text_file = io.BytesIO(b"Not an image")

    response = client.post(
        "/api/student_answers/3-Rizal_Seatwork-1/202160151/label_save_boxes",
        files={"files": ("test.txt", text_file, "text/plain")}
    )

    assert response.status_code == 415
    data = response.json()
    assert "detail" in data
    assert "unsupported" in data["detail"].lower()


def test_label_save_boxes_test_not_found():
    """Test POST /api/student_answers/{test_id}/{student_no}/label_save_boxes - Test not found"""
    test_image = create_test_image()

    response = client.post(
        "/api/student_answers/non-existent/202160151/label_save_boxes",
        files={"files": ("test.jpg", test_image, "image/jpeg")}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_commit_boxes_test_not_found():
    """Test POST /api/student_answers/{test_id}/{student_no}/commit_boxes - Test not found"""
    response = client.post(
        "/api/student_answers/non-existent/202160151/commit_boxes",
        json={"boxes": []}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_commit_boxes_student_not_found():
    """Test POST /api/student_answers/{test_id}/{student_no}/commit_boxes - Student not found"""
    response = client.post(
        "/api/student_answers/3-Rizal_Seatwork-1/999999999/commit_boxes",
        json={"boxes": []}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_update_answer_segmentation_invalid_points():
    """Test PATCH /api/student_answers/{test_id}/{student_no}/{item_id} - Invalid points format"""
    test_image = create_test_image()

    # Missing required corner points
    invalid_points = {
        "ul": {"x": 0, "y": 0}
        # Missing ur, lr, ll
    }

    response = client.patch(
        "/api/student_answers/3-Rizal_Seatwork-1/202160151/1",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"points": json.dumps(invalid_points)}
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "points" in data["detail"].lower()


def test_get_test_paper_statuses_success():
    """Test GET /api/student_answers/{test_id}/statuses - Success"""
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/statuses")

    assert response.status_code == 200
    data = response.json()

    assert data["test_id"] == "3-Rizal_Seatwork-1"
    assert "statuses" in data
    assert isinstance(data["statuses"], list)
    assert len(data["statuses"]) == 2

    status_item = data["statuses"][0]
    assert "student_no" in status_item
    assert "name" in status_item
    assert "is_done_rendering" in status_item
    assert "has_any_answer" in status_item
    assert "total_score" in status_item


def test_get_test_paper_statuses_not_found():
    """Test GET /api/student_answers/{test_id}/statuses - Not found"""
    response = client.get("/api/student_answers/non-existent/statuses")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_get_ai_evaluation_results_success():
    """Test GET /api/student_answers/{test_id}/results - Success"""
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/results")

    assert response.status_code == 200
    data = response.json()

    assert data["test_id"] == "3-Rizal_Seatwork-1"
    assert "students" in data
    assert isinstance(data["students"], list)
    assert len(data["students"]) == 2
    assert "student_no" in data["students"][0]
    assert "name" in data["students"][0]
    assert "evaluations" in data["students"][0]


def test_get_ai_evaluation_results_not_found():
    """Test GET /api/student_answers/{test_id}/results - Not found"""
    response = client.get("/api/student_answers/non-existent/results")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_get_ai_evaluation_results_per_student_success():
    """Test GET /api/student_answers/{test_id}/results/{student_no} - Success"""
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/results/202160151")

    assert response.status_code == 200
    data = response.json()

    assert data["test_id"] == "3-Rizal_Seatwork-1"
    assert data["student_no"] == "202160151"
    assert data["name"] == "Mohammad Hamdi Tuan"
    assert "evaluations" in data
    assert isinstance(data["evaluations"], list)


def test_get_ai_evaluation_results_per_student_test_not_found():
    """Test GET /api/student_answers/{test_id}/results/{student_no} - Test not found"""
    response = client.get("/api/student_answers/non-existent/results/202160151")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_get_ai_evaluation_results_per_student_student_not_found():
    """Test GET /api/student_answers/{test_id}/results/{student_no} - Student not found"""
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/results/999999999")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "student" in data["detail"].lower()


def test_get_ai_evaluation_results_per_student_wrong_section():
    """Test GET /api/student_answers/{test_id}/results/{student_no} - Student not in test's section"""
    # Student 202022222 is in section 2 (3-Aguinaldo), but test is for section 1 (3-Rizal)
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/results/202022222")

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_get_student_answers_success():
    """Test GET /api/student_answers/{test_id}/{student_no} - Success (no answers yet)"""
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/202160151")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_student_answers_test_not_found():
    """Test GET /api/student_answers/{test_id}/{student_no} - Test not found"""
    response = client.get("/api/student_answers/non-existent/202160151")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "test instance" in data["detail"].lower()


def test_get_student_answers_student_not_found():
    """Test GET /api/student_answers/{test_id}/{student_no} - Student not found"""
    response = client.get("/api/student_answers/3-Rizal_Seatwork-1/999999999")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "student" in data["detail"].lower()


def test_delete_student_answer_not_found():
    """Test DELETE /api/student_answers/{item_id}/{student_no} - No paper exists for student"""
    # No TestPaperInstance exists in mock data, so paper lookup returns 404
    response = client.delete("/api/student_answers/1/202160151")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
#endregion
