from fastapi import FastAPI, HTTPException, Response, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import os
from pathlib import Path
import json
import numpy as np
import cv2

from models import TestInstance, TestItem, Person, Section, TestPaperInstance
from schemas import (
    TestItemSummary,
    TestItemsResponse,
    NewTestItemRequest,
    NewTestItemResponse,
    UpdateTestItemRequest,
    FullTestItemResponse,
    NewTestInstanceRequest,
    UpdateTestInstanceRequest,
    TestInstanceResponse,
)
from database import create_db_and_tables, engine, get_direct_session
from fastapi import File, UploadFile, Form
from image_preprocessor import CVImagePreprocessor, CVProcessingError
from fastapi.responses import FileResponse
from sqlmodel import select

# ==============================
# Static Test Data (Temporary – Replace with DB Later)
# ==============================

TEST_INSTANCES = [
    {
        "name": "Seatwork-1",
        "section": "3-Rizal",
        "date": "2025-11-11T20:17:46.384Z",
        "test_id": "3-Rizal_Seatwork-1",
        "is_done_rendering": True
    },
    {
        "name": "Seatwork-2",
        "section": "3-Aguinaldo",
        "date": "2025-12-12T20:17:46.384Z",
        "test_id": "3-Aguinaldo_Seatwork-2",
        "is_done_rendering": False
    },
    {
        "name": "Quiz-1",
        "section": "3-Aguinaldo",
        "date": "2026-01-12T20:17:46.384Z",
        "test_id": "3-Aguinaldo_Quiz-1",
        "is_done_rendering": False
    },
    {
        "name": "Quiz-1",
        "section": "3-Rizal",
        "date": "2026-01-13T20:17:46.384Z",
        "test_id": "3-Rizal_Quiz-1",
        "is_done_rendering": True
    },
    {
        "name": "Quiz-2",
        "section": "3-Aguinaldo",
        "date": "2026-01-19T20:17:46.384Z",
        "test_id": "3-Aguinaldo_Quiz-2",
        "is_done_rendering": True
    }
]

# Use a set for O(1) lookup
VALID_TEST_IDS = {inst["test_id"] for inst in TEST_INSTANCES}

# Simulated in-memory storage for test items (to be replaced by DB)
test_items_db = {
    "3-Rizal_Seatwork-1": [
        {
            "item_id": "item_1",
            "question": "Solve for x: 2x + 5 = 15",
            "is_problem_solving": True,
            "expected_answer_rubric_questions": "Correct equation setup (2pts), accurate solution (2pts), proper units (1pt)"
        },
        {
            "item_id": "item_2",
            "question": "What is the capital of France?",
            "is_problem_solving": False,
            "expected_answer_rubric_questions": "Correct answer: Paris (1pt)"
        }
    ],
    "3-Aguinaldo_Quiz-2": [
        {
            "item_id": "item_1",
            "question": "Calculate the area of a circle with radius 7cm",
            "is_problem_solving": True,
            "expected_answer_rubric_questions": "Correct formula (2pts), substitution (1pt), calculation (1pt), units (1pt)"
        }
    ]
}

# ==============================
# Database Seeding (REMOVE AFTER TESTING)
# ==============================

create_db_and_tables()

from sqlmodel import Session as SQLModelSession
with SQLModelSession(engine) as session:
    # 1. Sections 
    for sec_name in ["3-Rizal", "3-Aguinaldo"]:
        if not session.exec(select(Section).where(Section.section_name == sec_name)).first():
            session.add(Section(section_name=sec_name))
    
    # 2. Students
    students_data = [
        # Section: 3-Rizal
        {"student_no": 202160151, "name": "Mohammad Hamdi S. Tuan", "section": "3-Rizal"},
        {"student_no": 202160152, "name": "Jose Ernesto Tomanan", "section": "3-Rizal"},
        {"student_no": 202160153, "name": "Bong Revilla", "section": "3-Rizal"},
        {"student_no": 202011111, "name": "Kean Baclaan", "section": "3-Rizal"},
        
        # Section: 3-Aguinaldo
        {"student_no": 202160154, "name": "Ana Manalang", "section": "3-Aguinaldo"},
        {"student_no": 202160155, "name": "Jose Rizal", "section": "3-Aguinaldo"},
        {"student_no": 202160156 , "name": "Pedro Penduko", "section": "3-Aguinaldo"},
    ]
    
    for student in students_data:
        if not session.exec(
            select(Person).where(Person.student_no == student["student_no"])
        ).first():
            session.add(Person(
                student_no=student["student_no"],
                name=student["name"],
                section=student["section"]
            ))
    
    # 3. Test Instances
    for inst in TEST_INSTANCES:
        if not session.exec(select(TestInstance).where(TestInstance.test_id == inst["test_id"])).first():
            session.add(TestInstance(
                test_id=inst["test_id"],
                name=inst["name"],
                section=inst["section"],
                date=inst["date"],
                is_done_rendering=inst["is_done_rendering"]
            ))
    
    # 4. Test Items (sync with in-memory test_items_db)
    for test_id, items in test_items_db.items():
        for item in items:
            if not session.exec(select(TestItem).where(TestItem.item_id == item["item_id"])).first():
                session.add(TestItem(
                    item_id=item["item_id"],
                    test_id=test_id,
                    question=item["question"],
                    is_problem_solving=item["is_problem_solving"],
                    expected_answer_rubric_questions=item["expected_answer_rubric_questions"]
                ))
    session.commit()

# ==============================
# App Setup
# ==============================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory for processed images
TEMP_DIR = Path("temp_cv_output")
TEMP_DIR.mkdir(exist_ok=True)

# ==============================
# Existing Endpoints (Unchanged)
# ==============================

@app.get("/api/test_instances")
def get_test_instances():
    return {"instances": TEST_INSTANCES}

@app.get(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def get_test_instance(test_id: str):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    instance = next((inst for inst in TEST_INSTANCES if inst["test_id"] == test_id), None)
    if instance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    return instance

@app.post(
    "/api/test_instances",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Missing required fields"},
        409: {"description": "Test instance already exists"}
    }
)
def add_test_instance(request: NewTestInstanceRequest):
    name = request.name
    section = request.section
    date = request.date
    
    if not name or not section or not date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: name, section, and date"
        )
    
    test_id = f"{section}_{name}"
    if test_id in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Test instance with ID '{test_id}' already exists"
        )
    
    new_instance = {
        "name": name,
        "section": section,
        "date": date,
        "test_id": test_id,
        "is_done_rendering": False
    }
    
    TEST_INSTANCES.append(new_instance)
    VALID_TEST_IDS.add(test_id)
    
    return new_instance

@app.patch(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format or update payload"},
        404: {"description": "Test instance not found"}
    }
)
def edit_test_instance(test_id: str, update: UpdateTestInstanceRequest):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if update.date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'date' can be updated for a test instance"
        )
    
    target_instance = None
    for inst in TEST_INSTANCES:
        if inst["test_id"] == test_id:
            inst["date"] = update.date
            target_instance = inst
            break
    
    items = test_items_db.get(test_id, [])
    summary_items = [
        TestItemSummary(
            question=item["question"],
            is_problem_solving=item["is_problem_solving"]
        )
        for item in items
    ]

    assert target_instance is not None

    return TestInstanceResponse(
        name=target_instance["name"],
        section=target_instance["section"],
        date=target_instance["date"],
        test_id=target_instance["test_id"],
        is_done_rendering=target_instance["is_done_rendering"],
        items=summary_items
    )

@app.delete(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def delete_test_instance(test_id: str):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    global TEST_INSTANCES
    TEST_INSTANCES = [inst for inst in TEST_INSTANCES if inst["test_id"] != test_id]
    VALID_TEST_IDS.discard(test_id)
    
    test_items_db.pop(test_id, None)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

def get_test_items(test_id: str):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    items = test_items_db.get(test_id, [])
    summary_items = [
        TestItemSummary(
            question=item["question"],
            is_problem_solving=item["is_problem_solving"]
        )
        for item in items
    ]
    
    return TestItemsResponse(test_id=test_id, items=summary_items)

@app.post(
    "/api/test_instances/{test_id}/items",
    response_model=NewTestItemResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input or duplicate item_id"},
        404: {"description": "Test instance not found"}
    }
)
def add_test_item(test_id: str, item: NewTestItemRequest):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if test_id not in test_items_db:
        test_items_db[test_id] = []
    
    if any(existing["item_id"] == item.item_id for existing in test_items_db[test_id]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Item ID '{item.item_id}' already exists in test instance '{test_id}'"
        )
    
    test_items_db[test_id].append(item.dict())
    return NewTestItemResponse(items=[item])

@app.patch(
    "/api/test_instances/{test_id}/{test_item_id}",
    response_model=FullTestItemResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid ID format"},
        404: {"description": "Test instance or item not found"}
    }
)
def edit_test_item(test_id: str, test_item_id: str, update: UpdateTestItemRequest):
    if not test_id or len(test_id) > 100 or not test_item_id or len(test_item_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if test_id not in test_items_db or not any(
        item["item_id"] == test_item_id for item in test_items_db[test_id]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test item with ID '{test_item_id}' not found in test instance '{test_id}'"
        )
    
    for item in test_items_db[test_id]:
        if item["item_id"] == test_item_id:
            update_data = update.dict(exclude_unset=True)
            item.update(update_data)
            return FullTestItemResponse(**item)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Item update failed unexpectedly"
    )

@app.delete(
    "/api/test_instances/{test_id}/{test_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid ID format"},
        404: {"description": "Test instance or item not found"}
    }
)
def delete_test_item(test_id: str, test_item_id: str):
    if not test_id or len(test_id) > 100 or not test_item_id or len(test_item_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if test_id not in test_items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No items found for test instance '{test_id}'"
        )
    
    original_count = len(test_items_db[test_id])
    test_items_db[test_id] = [
        item for item in test_items_db[test_id] if item["item_id"] != test_item_id
    ]
    
    if len(test_items_db[test_id]) == original_count:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test item with ID '{test_item_id}' not found in test instance '{test_id}'"
        )
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ==============================
# Computer Vision Preprocessing Endpoint (Multi-Box)
# ==============================

@app.post(
    "/api/test_instances/image_preprocess",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input (no file, empty file, or invalid format)"},
        415: {"description": "Unsupported media type (not JPEG/PNG)"},
        500: {"description": "CV processing failed"}
    }
)
async def image_preprocess(file: UploadFile = File(...)):
    """
    Process raw student assessment image through CV pipeline.
    Detects up to 3 document-like regions, applies brightening + contrast.
    Returns metadata with URLs to access each processed box.
    """
    # --- Validation ---
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    if not CVImagePreprocessor.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format. Allowed: .jpg, .jpeg, .png. Got: {file.filename}"
        )
    
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}"
        )

    # --- Processing ---
    try:
        preprocessor = CVImagePreprocessor()
        processed_list = preprocessor.process_assessment_image(contents)
    except CVProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV processing failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during image processing: {str(e)}"
        )

    # --- Save to temp files and generate URLs ---
    session_id = str(uuid.uuid4())
    boxes_info = []

    for i, img_bytes in enumerate(processed_list):
        filename = f"{session_id}_{i}.jpg"
        filepath = TEMP_DIR / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        boxes_info.append({
            "index": i,
            "image_path": f"/api/temp/{filename}"
        })

    return {
        "num_boxes": len(processed_list),
        "boxes": boxes_info
    }

# ==============================
# Test Paper Instance Endpoints (FULLY FIXED)
# ==============================

@app.post(
    "/api/test_instances/{test_id}/{student_no}/{item_id}/image_preprocess",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input format or student_no not integer"},
        404: {"description": "Test instance, student, or item not found"},
        415: {"description": "Unsupported media type"},
        500: {"description": "CV processing failed"}
    }
)
async def process_student_answer_image(
    test_id: str,
    student_no: str,
    item_id: str,
    file: UploadFile = File(...)
):
    # ===== VALIDATION PHASE =====
    try:
        student_no_int = int(student_no)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no must be a valid integer"
        )
    
    # Validate IDs exist in DATABASE
    session = get_direct_session()
    try:
        # Check test instance exists
        db_test = session.exec(
            select(TestInstance).where(TestInstance.test_id == test_id)
        ).first()
        if not db_test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found in database"
            )
        
        # Check student exists
        db_student = session.exec(
            select(Person).where(Person.student_no == student_no_int)
        ).first()
        if not db_student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
            )
        
        # Check item exists AND belongs to this test
        db_item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id,
                TestItem.test_id == test_id
            )
        ).first()
        if not db_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item '{item_id}' not found in test instance '{test_id}'"
            )
    finally:
        session.close()
    
    # Validate file
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    if not CVImagePreprocessor.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format. Allowed: .jpg, .jpeg, .png. Got: {file.filename}"
        )
    
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}"
        )
    
    # ===== PROCESSING PHASE =====
    try:
        preprocessor = CVImagePreprocessor()
        processed_list = preprocessor.process_assessment_image(contents)
    except CVProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV processing failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected processing error: {str(e)}"
        )
    
    # ===== STORAGE & RESPONSE PHASE =====
    session_id = str(uuid.uuid4())
    boxes_info = []
    
    for i, img_bytes in enumerate(processed_list):
        safe_filename = f"{test_id}_{student_no}_{item_id}_{session_id}_{i}.jpg"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        filepath = TEMP_DIR / safe_filename
        
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        
        boxes_info.append({
            "index": i,
            "image_directory": f"/api/temp/{safe_filename}"
        })
    
    return {
        "image_directory": session_id,
        "num_boxes": len(processed_list),
        "boxes": boxes_info
    }

@app.patch(
    "/api/test_instances/{test_id}/{student_no}/{item_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid points format, student_no not integer, or missing file"},
        404: {"description": "Test instance, student, or item not found"},
        415: {"description": "Unsupported media type"},
        500: {"description": "Image processing failed"}
    }
)
async def update_answer_segmentation(
    test_id: str,
    student_no: str,
    item_id: str,
    file: UploadFile = File(...),
    points: str = Form(...)
):
    # ===== VALIDATION PHASE =====
    try:
        student_no_int = int(student_no)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no must be a valid integer"
        )
    
    # Validate existence in DATABASE
    session = get_direct_session()
    try:
        if not session.exec(select(TestInstance).where(TestInstance.test_id == test_id)).first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Test instance '{test_id}' not found")
        if not session.exec(select(Person).where(Person.student_no == student_no_int)).first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student '{student_no}' not found")
        if not session.exec(
            select(TestItem).where(TestItem.item_id == item_id, TestItem.test_id == test_id)
        ).first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Item '{item_id}' not found in test '{test_id}'")
    finally:
        session.close()
    
    # Validate file
    if not file or not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image file provided")
    if not CVImagePreprocessor.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format: {file.filename}"
        )
    
    # Validate and parse points JSON
    try:
        points_data = json.loads(points)
        required = ["ul", "ur", "lr", "ll"]
        for corner in required:
            if corner not in points_data:
                raise ValueError(f"Missing corner point: {corner}")
            if not all(k in points_data[corner] for k in ["x", "y"]):
                raise ValueError(f"Point {corner} missing x/y coordinates")
            float(points_data[corner]["x"])
            float(points_data[corner]["y"])
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid points format: {str(e)}"
        )
    
    # ===== IMAGE PROCESSING PHASE =====
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image content")
        
        src_pts = np.array([
            [points_data["ul"]["x"], points_data["ul"]["y"]],
            [points_data["ur"]["x"], points_data["ur"]["y"]],
            [points_data["lr"]["x"], points_data["lr"]["y"]],
            [points_data["ll"]["x"], points_data["ll"]["y"]]
        ], dtype=np.float32)
        
        def calc_aspect(pts):
            w_top = np.linalg.norm(pts[0] - pts[1])
            w_bot = np.linalg.norm(pts[3] - pts[2])
            h_left = np.linalg.norm(pts[0] - pts[3])
            h_right = np.linalg.norm(pts[1] - pts[2])
            return (w_top + w_bot) / (h_left + h_right + 1e-5)
        
        aspect_ratio = calc_aspect(src_pts)
        OUT_HEIGHT = 800
        OUT_WIDTH = int(OUT_HEIGHT * aspect_ratio)
        
        dst_pts = np.float32([
            [0, 0],
            [OUT_WIDTH, 0],
            [OUT_WIDTH, OUT_HEIGHT],
            [0, OUT_HEIGHT]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (OUT_WIDTH, OUT_HEIGHT))
        
        preprocessor = CVImagePreprocessor()
        enhanced = preprocessor.brighten(warped, amount=0.2)
        enhanced = preprocessor.adjust_contrast(enhanced, amount=1.2)
        
        success, buffer = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise CVProcessingError("Failed to encode processed image")
        img_bytes = buffer.tobytes()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image segmentation failed: {str(e)}"
        )
    
    # ===== STORAGE & RESPONSE =====
    safe_filename = f"segmented_{test_id}_{student_no}_{item_id}_{uuid.uuid4().hex}.jpg"
    filepath = TEMP_DIR / safe_filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    
    return {
        "image_directory": f"/api/temp/{safe_filename}"
    }

@app.get(
    "/api/test_instances/{test_id}/statuses",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def get_test_paper_statuses(test_id: str):
    """Return per-student rendering status for a test instance (per updated contract)"""
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    session = get_direct_session()
    try:
        # Validate test instance exists
        test_inst = session.exec(
            select(TestInstance).where(TestInstance.test_id == test_id)
        ).first()
        if not test_inst:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
            )
        
        # Get all students in the test's section
        students = session.exec(
            select(Person).where(Person.section == test_inst.section)
        ).all()
        
        # Build contract-compliant response
        statuses = []
        for student in students:
            # Check if ALL items for this student have processed files
            items = session.exec(
                select(TestItem).where(TestItem.test_id == test_id)
            ).all()
            
            all_items_processed = True
            for item in items:
                # Check for ANY processed file (Endpoint 1 or Endpoint 2 pattern)
                pattern1 = f"{test_id}_{student.student_no}_{item.item_id}_"
                pattern2 = f"segmented_{test_id}_{student.student_no}_{item.item_id}_"
                if not any(
                    f.name.startswith(pattern1) or f.name.startswith(pattern2)
                    for f in TEMP_DIR.glob("*.jpg")
                ):
                    all_items_processed = False
                    break
            
            statuses.append({
                "student_no": str(student.student_no),  # Contract requires string
                "is_done_rendering": all_items_processed
            })
        
        return {
            "test_id": test_id,
            "statuses": statuses  # ONLY these two top-level fields per contract
        }
    finally:
        session.close()

@app.get(
    "/api/test_instances/{test_id}/results",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def get_ai_evaluation_results(test_id: str):
    """Return AI evaluations per updated contract (placeholder until AI module implemented)"""
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    session = get_direct_session()
    try:
        # Validate test instance exists
        if not session.exec(
            select(TestInstance).where(TestInstance.test_id == test_id)
        ).first():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
            )
        
        # Contract-compliant response structure (empty until AI implemented)
        return {
            "test_id": test_id,
            "evaluations": []  # Array of {student_no: str, ai_evaluation: str}
        }
    finally:
        session.close()

@app.get(
    "/api/test_instances/{test_id}/items",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def get_test_instance_items(test_id: str):
    """
    Get all test items for a specific test instance (per API contract).
    Returns items with item_id, label, question, and is_problem_solving.
    """
    # Validate test_id format
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    # Check if test instance exists in DATABASE
    session = get_direct_session()
    try:
        db_test = session.exec(
            select(TestInstance).where(TestInstance.test_id == test_id)
        ).first()
        
        if not db_test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
            )
        
        # Get all items for this test from DATABASE
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        # Format response per contract
        items_response = []
        for item in items:
            items_response.append({
                "item_id": item.item_id,
                "label": item.label if item.label else "",  # Handle None
                "question": item.question,
                "is_problem_solving": item.is_problem_solving
            })
        
        return {
            "test_id": test_id,
            "items": items_response
        }
    
    finally:
        session.close()


# ==============================
# NEW ENDPOINT 2(a): Get answers by student for test
# ==============================

@app.get(
    "/api/test_instances/{test_id}/{student_no}",
    status_code=status.HTTP_200_OK,
    responses={
        204: {"description": "Test and student exist but no answers found"},
        400: {"description": "Invalid test_id or student_no format"},
        404: {"description": "Test instance or student not found"}
    }
)
def get_student_answers(test_id: str, student_no: str):
    """
    Get all answers by a specific student for a test instance (per API contract).
    Returns answer metadata including image URLs and AI evaluation status.
    """
    # Validate test_id format
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    # Validate and convert student_no to integer
    try:
        student_no_int = int(student_no)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Check if test instance exists
        db_test = session.exec(
            select(TestInstance).where(TestInstance.test_id == test_id)
        ).first()
        
        if not db_test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
            )
        
        # Check if student exists
        db_student = session.exec(
            select(Person).where(Person.student_no == student_no_int)
        ).first()
        
        if not db_student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
            )
        
        # Get all items for this test
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        # Build answers array by checking for processed files
        answers = []
        for item in items:
            # Check for BOTH processing patterns:
            # 1. Endpoint 1 files: {test_id}_{student}_{item}_*.jpg
            # 2. Endpoint 2 files: segmented_{test_id}_{student}_{item}_*.jpg
            
            pattern1 = f"{test_id}_{student_no}_{item.item_id}_"
            pattern2 = f"segmented_{test_id}_{student_no}_{item.item_id}_"
            
            matching_files = [
                f for f in TEMP_DIR.glob("*.jpg")
                if f.name.startswith(pattern1) or f.name.startswith(pattern2)
            ]
            
            # If files exist, create answer entry
            if matching_files:
                # Use the most recent file (alphabetically last)
                latest_file = sorted(matching_files, reverse=True)[0]
                
                answers.append({
                    "answer_id": item.item_id,  # Using item_id as answer_id for now
                    "student_no": str(student_no),  # Contract requires string
                    "item_id": item.item_id,
                    "label": item.label if item.label else "",
                    "image_directory": f"/api/temp/{latest_file.name}",
                    "ai_evaluation": "",  # Placeholder - AI module not implemented yet
                    "is_done_rendering": True
                })
        
        # Return 204 No Content if no answers found
        if not answers:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        
        return {
            "answers": answers
        }
    
    finally:
        session.close()
        
# --- Serve temporary processed images ---
@app.get("/api/temp/{filename}")
async def get_processed_image(filename: str):
    if not filename.endswith(".jpg") or ".." in filename or not filename.replace(".jpg", "").replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = TEMP_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Processed image not found or expired")
    
    return FileResponse(filepath, media_type="image/jpeg")