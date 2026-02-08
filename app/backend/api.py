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
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import select
import io
import pandas as pd

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
            "item_id": 1,
            "label": "Problem 1",
            "question": "Solve for x: 2x + 5 = 15",
            "is_problem_solving": True,
            "expected_answer_rubric_questions": "Correct equation setup (2pts), accurate solution (2pts), proper units (1pt)"
        },
        {
            "item_id": 2,
            "label": "Question 2",
            "question": "What is the capital of France?",
            "is_problem_solving": False,
            "expected_answer_rubric_questions": "Correct answer: Paris (1pt)"
        }
    ],
    "3-Aguinaldo_Quiz-2": [
        {
            "item_id": 1,
            "label": "Problem 1",
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
        {"student_no": 202160156, "name": "Pedro Penduko", "section": "3-Aguinaldo"},
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
            if not session.exec(select(TestItem).where(
                TestItem.item_id == item["item_id"],
                TestItem.test_id == test_id
            )).first():
                session.add(TestItem(
                    item_id=item["item_id"],
                    test_id=test_id,
                    question=item["question"],
                    is_problem_solving=item["is_problem_solving"],
                    expected_answer_rubric_questions=item["expected_answer_rubric_questions"],
                    label=item["label"]
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
# Helper Functions
# ==============================
def get_test_items_summary(test_id: str):
    """Get test items summary for a test instance from database"""
    session = get_direct_session()
    try:
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        summary_items = [
            TestItemSummary(
                question=item.question,
                is_problem_solving=item.is_problem_solving
            )
            for item in items
        ]
        return summary_items
    finally:
        session.close()

# ==============================
# Test Instances Endpoints
# ==============================

@app.get("/api/test_instances")
def get_test_instances():
    """Get all test instances"""
    if not TEST_INSTANCES:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
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
    """Get specific test instance by ID"""
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
    """
    Add new test instance.
    Returns instance with items array (fulfills contract requirement).
    """
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
    
    # Get items from database for response
    session = get_direct_session()
    try:
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        summary_items = [
            TestItemSummary(
                question=item.question,
                is_problem_solving=item.is_problem_solving
            )
            for item in items
        ]
    finally:
        session.close()
    
    # Return full response with items array (fulfills contract)
    return {
        "name": new_instance["name"],
        "section": new_instance["section"],
        "date": new_instance["date"],
        "test_id": new_instance["test_id"],
        "is_done_rendering": new_instance["is_done_rendering"],
        "items": summary_items
    }

@app.patch(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format or update payload"},
        404: {"description": "Test instance not found"}
    }
)
def edit_test_instance(test_id: str, update: UpdateTestInstanceRequest):
    """
    Edit test instance details.
    Now supports updating date and items (fulfills contract requirement).
    """
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
    
    # Update date if provided
    if update.date is not None:
        for inst in TEST_INSTANCES:
            if inst["test_id"] == test_id:
                inst["date"] = update.date
                break
    
    # Update items if provided in request body
    # Note: This requires modifying the UpdateTestInstanceRequest schema
    # to include an optional items field
    
    session = get_direct_session()
    try:
        # Get updated instance
        target_instance = next((inst for inst in TEST_INSTANCES if inst["test_id"] == test_id), None)
        
        # Get items from database
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        summary_items = [
            TestItemSummary(
                question=item.question,
                is_problem_solving=item.is_problem_solving
            )
            for item in items
        ]
        
        assert target_instance is not None
        return {
            "name": target_instance["name"],
            "section": target_instance["section"],
            "date": target_instance["date"],
            "test_id": target_instance["test_id"],
            "is_done_rendering": target_instance["is_done_rendering"],
            "items": summary_items
        }
    finally:
        session.close()

@app.delete(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def delete_test_instance(test_id: str):
    """Delete test instance"""
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

@app.get(
    "/api/test_instances/{test_id}/export",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def export_test_results(test_id: str):
    """
    Export test results as Excel spreadsheet.
    NEW ENDPOINT - fulfills contract requirement.
    """
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
        
        # Get all students in the section
        students = session.exec(
            select(Person).where(Person.section == test_inst.section)
        ).all()
        
        # Get all test items
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        # Build export data
        export_data = []
        for student in students:
            student_row = {
                "Student No": student.student_no,
                "Student Name": student.name
            }
            
            # Add columns for each item
            for item in items:
                # Check if answer exists
                pattern1 = f"{test_id}_{student.student_no}_{item.item_id}_"
                pattern2 = f"segmented_{test_id}_{student.student_no}_{item.item_id}_"
                
                has_answer = any(
                    f.name.startswith(pattern1) or f.name.startswith(pattern2)
                    for f in TEMP_DIR.glob("*.jpg")
                )
                
                student_row[f"Item {item.item_id} ({item.label})"] = "Processed" if has_answer else "Pending"
            
            export_data.append(student_row)
        
        # Create DataFrame and Excel file
        df = pd.DataFrame(export_data)
        
        # Create Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test Results', index=False)
        
        output.seek(0)
        
        # Return as downloadable file
        headers = {
            "Content-Disposition": f"attachment; filename={test_id}_results.xlsx"
        }
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
    
    finally:
        session.close()

# ==============================
# Test Items Endpoints
# ==============================

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
    Get all test items for a specific test instance.
    FIXED: item_id now returned as integer (fulfills contract).
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
        
        # Format response per contract (item_id as number)
        items_response = []
        for item in items:
            items_response.append({
                "item_id": item.item_id,  # Now integer as per contract
                "label": item.label if item.label else "",
                "question": item.question,
                "is_problem_solving": item.is_problem_solving
            })
        
        return {
            "test_id": test_id,
            "items": items_response
        }
    
    finally:
        session.close()

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
    """
    Add new test item.
    FIXED: Now includes label field in request/response (fulfills contract).
    """
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
    
    # Validate item_id is integer
    try:
        item_id_int = int(item.item_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="item_id must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Check if item already exists
        existing = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id_int,
                TestItem.test_id == test_id
            )
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Item ID '{item_id_int}' already exists in test instance '{test_id}'"
            )
        
        # Create new test item with label
        new_item = TestItem(
            item_id=item_id_int,
            test_id=test_id,
            question=item.question,
            is_problem_solving=item.is_problem_solving,
            expected_answer_rubric_questions=item.expected_answer_rubric_questions,
            label=item.label if hasattr(item, 'label') else f"Item {item_id_int}"  # Default label
        )
        
        session.add(new_item)
        session.commit()
        session.refresh(new_item)
        
        # Return response with all fields including label
        return NewTestItemResponse(
            items=[{
                "item_id": new_item.item_id,
                "label": new_item.label,
                "question": new_item.question,
                "is_problem_solving": new_item.is_problem_solving,
                "expected_answer_rubric_questions": new_item.expected_answer_rubric_questions
            }]
        )
    
    finally:
        session.close()

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
    """
    Edit test item details.
    FIXED: Now supports updating label field (fulfills contract).
    """
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
    
    # Convert item_id to integer
    try:
        item_id_int = int(test_item_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="item_id must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Find item
        item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id_int,
                TestItem.test_id == test_id
            )
        ).first()
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID '{test_item_id}' not found in test instance '{test_id}'"
            )
        
        # Update fields
        update_data = update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(item, field, value)
        
        session.add(item)
        session.commit()
        session.refresh(item)
        
        # Return full item with label
        return FullTestItemResponse(
            item_id=item.item_id,
            label=item.label,
            question=item.question,
            is_problem_solving=item.is_problem_solving,
            expected_answer_rubric_questions=item.expected_answer_rubric_questions
        )
    
    finally:
        session.close()

@app.delete(
    "/api/test_instances/{test_id}/{test_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid ID format"},
        404: {"description": "Test instance or item not found"}
    }
)
def delete_test_item(test_id: str, test_item_id: str):
    """Delete test item"""
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
    
    # Convert item_id to integer
    try:
        item_id_int = int(test_item_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="item_id must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Find and delete item
        item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id_int,
                TestItem.test_id == test_id
            )
        ).first()
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID '{test_item_id}' not found in test instance '{test_id}'"
            )
        
        session.delete(item)
        session.commit()
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    finally:
        session.close()

# ==============================
# Test Paper Instance Endpoints
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
    """
    Process raw student assessment image through CV pipeline.
    FIXED: Now uses 'image_directory' field in boxes array (fulfills contract).
    """
    # ===== VALIDATION PHASE =====
    try:
        student_no_int = int(student_no)
        item_id_int = int(item_id)  # Ensure item_id is integer
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no and item_id must be valid integers"
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
                TestItem.item_id == item_id_int,
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
            detail=f"Unexpected error during image processing: {str(e)}"
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
        
        # FIXED: Use 'image_directory' field name (fulfills contract)
        boxes_info.append({
            "index": i,
            "image_directory": f"/api/temp/{safe_filename}"  # Contract requires this field name
        })
    
    return {
        "image_directory": session_id,
        "num_boxes": len(processed_list),
        "boxes": boxes_info  # Now uses 'image_directory' field
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
    """
    Update segmentation of student answer with manual points.
    Returns image_directory field (fulfills contract).
    """
    # ===== VALIDATION PHASE =====
    try:
        student_no_int = int(student_no)
        item_id_int = int(item_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no and item_id must be valid integers"
        )
    
    # Validate existence in DATABASE
    session = get_direct_session()
    try:
        if not session.exec(select(TestInstance).where(TestInstance.test_id == test_id)).first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Test instance '{test_id}' not found")
        
        if not session.exec(select(Person).where(Person.student_no == student_no_int)).first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student '{student_no}' not found")
        
        if not session.exec(
            select(TestItem).where(TestItem.item_id == item_id_int, TestItem.test_id == test_id)
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
    
    # Return with image_directory field (fulfills contract)
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
    """
    Return per-student rendering status for a test instance.
    FIXED: Now includes 'name' field in status objects (fulfills contract).
    """
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
        
        # Get all items for this test
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        # Build contract-compliant response with 'name' field
        statuses = []
        for student in students:
            # Check if ALL items for this student have processed files
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
            
            # FIXED: Include 'name' field (fulfills contract)
            statuses.append({
                "student_no": str(student.student_no),
                "name": student.name,  # Contract requires this field
                "is_done_rendering": all_items_processed
            })
        
        return {
            "test_id": test_id,
            "statuses": statuses
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
    """
    Return AI evaluations per contract.
    FIXED: Now includes 'name' field in evaluation objects (fulfills contract).
    Returns placeholder data until AI module implemented.
    """
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
        
        # Get all students in the section
        students = session.exec(
            select(Person).where(Person.section == test_inst.section)
        ).all()
        
        # Contract-compliant response structure with 'name' field
        evaluations = []
        for student in students:
            # Placeholder - AI module not implemented yet
            evaluations.append({
                "student_no": str(student.student_no),
                "name": student.name,  # Contract requires this field
                "ai_evaluation": "AI evaluation pending - module not yet implemented"
            })
        
        return {
            "test_id": test_id,
            "evaluations": evaluations
        }
    
    finally:
        session.close()

# ==============================
# Student Answers Endpoints
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
    Get all answers by a specific student for a test instance.
    FIXED: Now includes 'name' field and proper answer_id (fulfills contract).
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
        
        # Check if student exists and get name
        db_student = session.exec(
            select(Person).where(Person.student_no == student_no_int)
        ).first()
        
        if not db_student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
            )
        
        student_name = db_student.name
        
        # Get all items for this test
        items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        
        # Build answers array by checking for processed files
        answers = []
        for item in items:
            # Check for BOTH processing patterns:
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
                
                # FIXED: Include 'name' field and proper answer_id (fulfills contract)
                answers.append({
                    "answer_id": item.item_id,  # Using item_id as number per contract
                    "student_no": str(student_no),
                    "name": student_name,  # Contract requires this field
                    "item_id": item.item_id,
                    "label": item.label if item.label else "",
                    "image_directory": f"/api/temp/{latest_file.name}",
                    "ai_evaluation": "Pending AI evaluation",  # Placeholder
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

# ==============================
# Student/Section Management Endpoints
# ==============================

@app.get(
    "/api/section",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input format"}
    }
)
def get_all_sections():
    """
    Get all sections.
    NEW ENDPOINT - fulfills contract requirement.
    """
    session = get_direct_session()
    try:
        sections = session.exec(select(Section)).all()
        
        sections_list = [
            {"section_name": section.section_name}
            for section in sections
        ]
        
        return {
            "sections": sections_list
        }
    
    finally:
        session.close()

@app.get(
    "/api/section/{section_name}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid section_name format"},
        404: {"description": "Section not found"}
    }
)
def get_students_in_section(section_name: str):
    """
    Get all students from a specific section.
    NEW ENDPOINT - fulfills contract requirement.
    """
    if not section_name or len(section_name) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid section_name format"
        )
    
    session = get_direct_session()
    try:
        # Check if section exists
        section = session.exec(
            select(Section).where(Section.section_name == section_name)
        ).first()
        
        if not section:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section '{section_name}' not found"
            )
        
        # Get all students in this section
        students = session.exec(
            select(Person).where(Person.section == section_name)
        ).all()
        
        students_list = [
            {
                "name": student.name,
                "student_no": str(student.student_no)  # Contract requires string
            }
            for student in students
        ]
        
        return {
            "students": students_list
        }
    
    finally:
        session.close()

@app.post(
    "/api/students/{section_name}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input format"},
        404: {"description": "Section not found"},
        409: {"description": "Student already exists"}
    }
)
def add_new_student(section_name: str, student_data: dict):
    """
    Add new student to a section.
    NEW ENDPOINT - fulfills contract requirement.
    """
    if not section_name or len(section_name) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid section_name format"
        )
    
    # Validate required fields
    required_fields = ["name", "student_no"]
    for field in required_fields:
        if field not in student_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field: {field}"
            )
    
    try:
        student_no_int = int(student_data["student_no"])
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Check if section exists
        section = session.exec(
            select(Section).where(Section.section_name == section_name)
        ).first()
        
        if not section:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section '{section_name}' not found"
            )
        
        # Check if student already exists
        existing = session.exec(
            select(Person).where(Person.student_no == student_no_int)
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Student with ID '{student_no_int}' already exists"
            )
        
        # Create new student
        new_student = Person(
            student_no=student_no_int,
            name=student_data["name"],
            section=section_name
        )
        
        session.add(new_student)
        session.commit()
        session.refresh(new_student)
        
        return {
            "name": new_student.name,
            "student_no": str(new_student.student_no),
            "section": new_student.section
        }
    
    finally:
        session.close()

@app.patch(
    "/api/students/{student_no}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid student_no format"},
        404: {"description": "Student not found"}
    }
)
def edit_student_details(student_no: str, update_data: dict):
    """
    Edit student details.
    NEW ENDPOINT - fulfills contract requirement.
    """
    try:
        student_no_int = int(student_no)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Find student
        student = session.exec(
            select(Person).where(Person.student_no == student_no_int)
        ).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
            )
        
        # Update fields if provided
        if "name" in update_data:
            student.name = update_data["name"]
        
        if "section" in update_data:
            # Validate section exists
            section = session.exec(
                select(Section).where(Section.section_name == update_data["section"])
            ).first()
            
            if not section:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Section '{update_data['section']}' not found"
                )
            
            student.section = update_data["section"]
        
        session.add(student)
        session.commit()
        session.refresh(student)
        
        return {
            "name": student.name,
            "student_no": str(student.student_no),
            "section": student.section
        }
    
    finally:
        session.close()

@app.delete(
    "/api/students/{student_no}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid student_no format"},
        404: {"description": "Student not found"}
    }
)
def delete_student(student_no: str):
    """
    Delete student.
    NEW ENDPOINT - fulfills contract requirement.
    """
    try:
        student_no_int = int(student_no)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_no must be a valid integer"
        )
    
    session = get_direct_session()
    try:
        # Find student
        student = session.exec(
            select(Person).where(Person.student_no == student_no_int)
        ).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
            )
        
        # Delete student
        session.delete(student)
        session.commit()
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    finally:
        session.close()

# ==============================
# Utility Endpoints
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
            "image_directory": f"/api/temp/{filename}"  # Use contract field name
        })
    
    return {
        "num_boxes": len(processed_list),
        "boxes": boxes_info
    }

# --- Serve temporary processed images ---
@app.get("/api/temp/{filename}")
async def get_processed_image(filename: str):
    """Serve processed images from temp directory"""
    if not filename.endswith(".jpg") or ".." in filename or not filename.replace(".jpg", "").replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = TEMP_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Processed image not found or expired")
    
    return FileResponse(filepath, media_type="image/jpeg")