from fastapi import FastAPI, HTTPException, Response, status, Depends, File, UploadFile, Form, Body, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, delete, select
from typing import List, Optional

import uuid
import json
import numpy as np
import cv2
import io
import re
import pandas as pd
from pathlib import Path
from pydantic import ValidationError

from models import *
from schemas import *
from database import create_db_and_tables, get_session, engine, ENVIRONMENT
from functionality.image_preprocessor import CVImagePreprocessor, CVProcessingError
from functionality.ai_interface import *
from functionality.sheets_exporter import *

import logging
logger = logging.getLogger(__name__)


# ==============================
#region Preprocessor Configuration
# ==============================
USE_PADDLE_OCR = True  # Set to False to disable PaddleOCR and use traditional CV only
PADDLE_OCR_LANG = 'en'  # Language for PaddleOCR ('en', 'ch', 'fr', etc.)

#endregion



# ==============================
#region App Initialization
# ==============================
app = FastAPI(
        title="Assessment Processing API",
        description="API for managing test instances, items, and student answer processing",
        # lifespan=create_db_and_tables(),
        version="1.0.0"
        )

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )

AI_ANSWER_EVALUATOR = AIAnswerEvaluator()
CV_IMAGE_PREPROCESSOR = CVImagePreprocessor()

TEMP_DIR = Path("temp_cv_output")
TEMP_DIR.mkdir(exist_ok=True)

#endregion



# ==============================
#region AUTO-SEEDING ON STARTUP (DEV ONLY)
# ==============================
@app.on_event("startup")
async def startup_database_setup():
    """Create tables + conditionally seed dev DB"""
    # Always create tables first
    create_db_and_tables()
    
    # ONLY seed if explicitly enabled in development
    if ENVIRONMENT == "development" and os.getenv("AUTO_SEED", "false").lower() == "true":
        print(f"\nAUTO_SEED ENABLED (ENV={ENVIRONMENT}) - Resetting database with mock data...")
        try:
            # Import locally to avoid circular dependencies
            from functionality.seed_dev_db import seed_dev_database
            seed_dev_database()  # Uses safety checks from seed_dev_db.py
            print("Database seeded successfully on startup\n")
        except Exception as e:
            print(f"SEEDING FAILED ON STARTUP: {type(e).__name__}: {e}\n")
            # Fail fast since AUTO_SEED was explicitly requested
            raise RuntimeError("Critical: Database seeding failed during startup") from e
#endregion



# ==============================
#region Section Endpoints
# ==============================
@app.get("/api/sections", response_model=List[dict])
def get_all_sections(session: Session = Depends(get_session)):
    """Get all sections"""
    sections = session.exec(select(Section)).all()
    return [{"section_id": s.section_id, "section_name": s.section} for s in sections]


@app.get("/api/sections/{section_id}", response_model=List[StudentResponse])
def get_students_in_section(section_id: int, session: Session = Depends(get_session)):
    """Get all students from a specific section"""
    # Verify section exists
    section = session.get(Section, section_id)
    if not section:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {section_id} not found"
                )
    
    # Get students in this section
    students = session.exec(
            select(Student).where(Student.section_id == section_id)
            ).all()
    
    return [
        StudentResponse(
            student_no=s.student_no,
            name=s.name,
            section_id=s.section_id
        ) for s in students
        ]


@app.post("/api/students", status_code=status.HTTP_201_CREATED)
def add_new_student(
            student_data: dict = Body(...),
            session: Session = Depends(get_session)
            ):
    """Add new student (section_id provided in body per contract)"""
    # Validate required fields per contract
    required_fields = ["name", "student_no", "section_id"]
    for field in required_fields:
        if field not in student_data:
            raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                    )
    
    # Verify section exists FIRST (critical FK validation)
    section = session.get(Section, student_data["section_id"])
    if not section:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {student_data['section_id']} not found"
                )
    
    # Check for duplicate student ID
    existing = session.exec(
            select(Student).where(Student.student_no == student_data["student_no"])
            ).first()
    if existing:
        raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Student with ID '{student_data['student_no']}' already exists"
                )
    
    # Create student with section_id from body (NOT path param)
    new_student = Student(
            student_no=student_data["student_no"],
            name=student_data["name"],
            section_id=student_data["section_id"]  # Directly from body
            )
    session.add(new_student)
    session.commit()
    session.refresh(new_student)
    
    return StudentResponse(
            student_no=new_student.student_no,
            name=new_student.name,
            section_id=new_student.section_id,
            )


@app.patch("/api/students/{student_no}", response_model=StudentResponse)
def edit_student_details(
        student_no: str,
        update_data: dict,  # Parameter name matches usage below
        session: Session = Depends(get_session)
        ):
    """Edit student details"""
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
                )
    
    # Update fields if provided (FIXED: using update_data consistently)
    if "name" in update_data:  # CORRECTED: was update_
        student.name = update_data["name"]
    
    if "section_id" in update_data:  # CORRECTED: was update_
        # Verify section exists
        section = session.get(Section, update_data["section_id"])
        if not section:
            raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Section with ID {update_data['section_id']} not found"
                    )
        student.section_id = update_data["section_id"]
    
    session.add(student)
    session.commit()
    session.refresh(student)
    
    return StudentResponse(
            student_no=student.student_no,
            name=student.name,
            section_id=student.section_id
            )


@app.delete("/api/students/{student_no}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(student_no: str, session: Session = Depends(get_session)):
    """Delete student"""
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
                )
    
    session.delete(student)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
#endregion



# ==============================
#region Test Instance Endpoints
# ==============================
@app.get("/api/test_instances", response_model=List[TestInstanceResponse])
def get_test_instances(session: Session = Depends(get_session)):
    """Get all test instances with their items"""
    test_instances = session.exec(select(TestInstance)).all()
    
    if not test_instances:
        return []
    
    responses = []
    for instance in test_instances:
        items = session.exec(
                select(TestItem).where(TestItem.test_id == instance.test_id)
                ).all()
        
        summary_items = [
                TestItemSummary(
                    item_id=item.item_id,
                    label=item.label,
                    question=item.question,
                    is_problem_solving=item.is_problem_solving,
                    expected_answer_rubric_questions=item.expected_answer_rubric_questions
                ) for item in items
                ]
        
        responses.append(
                TestInstanceResponse(
                    name=instance.name,
                    section_id=instance.section_id,
                    date=instance.date,
                    test_id=instance.test_id,
                    is_done_rendering=instance.is_done_rendering,
                    items=summary_items
                )
                )
    
    return responses


@app.get("/api/test_instances/{test_id}", response_model=TestInstanceResponse)
def get_test_instance(test_id: str, session: Session = Depends(get_session)):
    """Get specific test instance by ID with items"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    
    summary_items = [
            TestItemSummary(
                item_id=item.item_id,
                label=item.label,
                question=item.question,
                is_problem_solving=item.is_problem_solving,
                expected_answer_rubric_questions=item.expected_answer_rubric_questions
            ) for item in items
            ]
    
    return TestInstanceResponse(
            name=instance.name,
            section_id=instance.section_id,
            date=instance.date,
            test_id=instance.test_id,
            is_done_rendering=instance.is_done_rendering,
            items=summary_items
            )


@app.post("/api/test_instances", response_model=TestInstanceResponse, status_code=status.HTTP_201_CREATED)
def add_test_instance(
            request: NewTestInstanceRequest,
            session: Session = Depends(get_session)
            ):
    """Add new test instance"""
    # Verify section exists
    section = session.get(Section, request.section_id)
    if not section:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {request.section_id} not found"
                )
    
    # Generate test_id (section-based naming)
    test_id = f"{section.section}_{request.name}_{uuid.uuid4().hex[:6]}"
    
    # Create new instance
    new_instance = TestInstance(
            test_id=test_id,
            name=request.name,
            section_id=request.section_id,
            date=request.date,
            is_done_rendering=False
            )
    session.add(new_instance)
    session.commit()
    session.refresh(new_instance)
    
    # Return with empty items array (no items created yet)
    return TestInstanceResponse(
            name=new_instance.name,
            section_id=new_instance.section_id,
            date=new_instance.date,
            test_id=new_instance.test_id,
            is_done_rendering=new_instance.is_done_rendering,
            items=[]
            )


@app.patch("/api/test_instances/{test_id}", response_model=TestInstanceResponse)
def edit_test_instance(
            test_id: str,
            update: UpdateTestInstanceRequest,
            session: Session = Depends(get_session)
            ):
    """Edit test instance details including date and items"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    # Update date if provided
    if update.date is not None:
        instance.date = update.date
    
    # Update items if provided (REPLACE all items)
    if update.items is not None:
        # Delete existing items
        existing_items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        for item in existing_items:
            session.delete(item)
        
        # Create new items
        for idx, item_data in enumerate(update.items, start=1):
            question = item_data.question if item_data.question is not None else "Untitled Question"
            is_problem_solving = item_data.is_problem_solving if item_data.is_problem_solving is not None else False

            new_item = TestItem(
                item_id=idx,
                test_id=test_id,
                label=item_data.label or f"Item {idx}",
                question=question,
                is_problem_solving=is_problem_solving,
                expected_answer_rubric_questions=item_data.expected_answer_rubric_questions or ""
            )
            session.add(new_item)
    
    session.add(instance)
    session.commit()
    session.refresh(instance)
    
    # Get updated items
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    
    summary_items = [
            TestItemSummary(
                item_id=item.item_id,
                label=item.label,
                question=item.question,
                is_problem_solving=item.is_problem_solving,
                expected_answer_rubric_questions=item.expected_answer_rubric_questions
            ) for item in items
            ]
    
    return TestInstanceResponse(
            name=instance.name,
            section_id=instance.section_id,
            date=instance.date,
            test_id=instance.test_id,
            is_done_rendering=instance.is_done_rendering,
            items=summary_items
            )


@app.delete("/api/test_instances/{test_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_test_instance(test_id: str, session: Session = Depends(get_session)):
    """Delete test instance and all related data"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    # Delete related test items
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    for item in items:
        session.delete(item)
    
    # Delete related test papers
    papers = session.exec(
            select(TestPaperInstance).where(TestPaperInstance.test_id == test_id)
            ).all()
    for paper in papers:
        # Delete related student answers first
        answers = session.exec(
                select(StudentAnswer).where(StudentAnswer.paper_id == paper.paper_id)
                ).all()
        for answer in answers:
            session.delete(answer)
        session.delete(paper)
    
    # Delete the instance itself
    session.delete(instance)
    session.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/test_instances/{test_id}/export")
def export_test_results(test_id: str, session: Session = Depends(get_session)):
    """Export test results as Excel spreadsheet"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Get all students in the section
    students = session.exec(
            select(Student).where(Student.section_id == instance.section_id)
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
            # Check if answer exists in database
            answer = session.exec(
                    select(StudentAnswer)
                    .join(TestPaperInstance)
                    .where(
                        TestPaperInstance.test_id == test_id,
                        TestPaperInstance.student_no == student.student_no,
                        StudentAnswer.item_id == item.item_id
                    )
                    ).first()
            
            student_row[f"Item {item.item_id} ({item.label})"] = (
                    "Processed" if answer and answer.is_done_rendering else "Pending"
                    )
        
        export_data.append(student_row)
    
    # Create DataFrame and Excel file
    df = pd.DataFrame(export_data)
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
#endregion



# ==============================
#region Test Item Endpoints
# ==============================
# @app.get("/api/test_instances/{test_id}/items", response_model=TestItemsResponse, responses={ 404: {"description": "Test instance not found"},})
# def get_test_instance_items(
#             test_id: str,
#             session: Session = Depends(get_session),
#             ):
#     """Get all test items for a specific test instance"""
#     instance = session.get(TestInstance, test_id)
#     if not instance:
#         raise HTTPException(status_code=404, detail=f"Test instance '{test_id}' not found")
    
#     items = session.exec(
#         select(TestItem).where(TestItem.test_id == test_id)
#     ).all()

#     items_summary = [
#             TestItemSummary(
#                 item_id=item.item_id,
#                 label=item.label,
#                 question=item.question,
#                 is_problem_solving=item.is_problem_solving,
#                 expected_answer_rubric_questions=item.expected_answer_rubric_questions,
#                 )
#             for item in items
#             ]

#     return TestItemsResponse(test_id=test_id, items=items_summary)
@app.get("/api/test_instances/{test_id}/items")
def get_test_instance_items(
    test_id: str,
    session: Session = Depends(get_session),
):
    """Get all test items for a specific test instance"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance '{test_id}' not found"
        )
    
    # Get items
    items = session.exec(
        select(TestItem).where(TestItem.test_id == test_id)
    ).all()
    
    # Build items list
    items_list = [
        {
            "item_id": item.item_id,
            "label": item.label,
            "question": item.question,
            "is_problem_solving": item.is_problem_solving,
            "expected_answer_rubric_questions": item.expected_answer_rubric_questions,
        }
        for item in items
    ]
    
    return {
        "test_id": test_id,
        "items": items_list
    }

@app.post("/api/test_instances/{test_id}/items", response_model=NewTestItemResponse)
def add_test_item(
            test_id: str,
            item: NewTestItemRequest,
            session: Session = Depends(get_session)
            ):
    """Add new test item (item_id auto-generated)"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Determine next item_id (max existing + 1)
    max_item = session.exec(
        select(TestItem)
        .where(TestItem.test_id == test_id)
        .order_by(TestItem.item_id.desc())  # type: ignore[union-attr]
    ).first()

    next_item_id = (max_item.item_id + 1) if max_item else 1

    # Create new test item
    question = item.question if item.question is not None else "Untitled Question"
    is_problem_solving = item.is_problem_solving if item.is_problem_solving is not None else False

    new_item = TestItem(
        item_id=next_item_id,
        test_id=test_id,
        label=item.label or f"Item {next_item_id}",
        question=question,
        is_problem_solving=is_problem_solving,
        expected_answer_rubric_questions=item.expected_answer_rubric_questions or ""
    )
    session.add(new_item)
    session.commit()
    session.refresh(new_item)
    
    # Return full item representation
    return NewTestItemResponse(
            items=[
                FullTestItemResponse(
                    item_id=new_item.item_id,
                    label=new_item.label,
                    question=new_item.question,
                    is_problem_solving=new_item.is_problem_solving,
                    expected_answer_rubric_questions=new_item.expected_answer_rubric_questions
                )
            ])


@app.patch("/api/test_instances/{test_id}/items/{item_id}", response_model=FullTestItemResponse)
def edit_test_item(
            test_id: str,
            item_id: int,
            update: UpdateTestItemRequest,
            session: Session = Depends(get_session)
            ):
    """Edit test item details"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Find item
    item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id,
                TestItem.test_id == test_id
            )
            ).first()
    
    if not item:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID {item_id} not found in test instance '{test_id}'"
                )
    
    # Update fields
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(item, field, value)
    
    session.add(item)
    session.commit()
    session.refresh(item)
    
    return FullTestItemResponse(
            item_id=item.item_id,
            label=item.label,
            question=item.question,
            is_problem_solving=item.is_problem_solving,
            expected_answer_rubric_questions=item.expected_answer_rubric_questions
            )


@app.delete("/api/test_instances/{test_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_test_item(
            test_id: str,
            item_id: int,
            session: Session = Depends(get_session)
            ):
    """Delete test item"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Find and delete item
    item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id,
                TestItem.test_id == test_id
                )
            ).first()
    
    if not item:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID {item_id} not found in test instance '{test_id}'"
                )
    
    # Delete related student answers first
    answers = session.exec(
            select(StudentAnswer)
            .join(TestPaperInstance)
            .where(
                TestPaperInstance.test_id == test_id,
                StudentAnswer.item_id == item_id
            )
            ).all()
    
    for answer in answers:
        session.delete(answer)
    
    session.delete(item)
    session.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)
#endregion



# ==============================
#region Student Answer Processing Endpoints
# ==============================


@app.post("/api/test_instances/{test_id}/{student_no}/{item_id}/image_preprocess")
async def process_student_answer_image(
    test_id: str,
    student_no: str,
    item_id: int,
    file: UploadFile = File(...),
    num_boxes: Optional[int] = Query(None), 
    session: Session = Depends(get_session)
):
    """Process raw student assessment image through CV pipeline"""
    # ===== VALIDATION =====
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance '{test_id}' not found"
        )
    # Verify student exists
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID '{student_no}' not found"
        )
    # Verify item exists and belongs to this test
    item = session.exec(
        select(TestItem).where(
            TestItem.item_id == item_id,
            TestItem.test_id == test_id
        )
    ).first()
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found in test instance '{test_id}'"
        )
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

    # ===== PROCESSING =====
    try:
        preprocessor = CVImagePreprocessor(
            use_paddle_ocr=USE_PADDLE_OCR,
            paddle_ocr_lang=PADDLE_OCR_LANG,
            debug_mode=False
        )
        if num_boxes is not None:
            processed_list = preprocessor.process_assessment_image(contents, num_boxes=num_boxes)
        else:
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

    # ===== SAVE ALL CANDIDATE BOXES FOR PREVIEW =====
    # Create or update TestPaperInstance (metadata only, no specific answer linked yet)
    paper = session.exec(
        select(TestPaperInstance).where(
            TestPaperInstance.test_id == test_id,
            TestPaperInstance.student_no == student_no
        )
    ).first()
    if not paper:
        paper = TestPaperInstance(
            test_id=test_id,
            student_no=student_no,
            is_done_rendering=False
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)
    
    assert paper is not None

    boxes_info = []
    default_image_dir = ""

    for i, img_bytes in enumerate(processed_list):
        # Generate filename
        safe_filename = f"{test_id}_{student_no}_{item_id}_{uuid.uuid4().hex}_{i}.jpg"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        filepath = TEMP_DIR / safe_filename
        
        # Save image file
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        
        image_dir = f"/api/temp/{safe_filename}"
        
        # Set the first box as the default selected image_directory
        if i == 0:
            default_image_dir = image_dir

            # Extract item number using AI
            item_number = "UNKNOWN"
            try:
                item_number = AI_ANSWER_EVALUATOR.get_item_number(img_bytes)
                if not item_number or item_number.strip() == "NONE":
                    item_number = "UNKNOWN"
                else:
                    item_number = item_number.strip()
            except Exception as e:
                logger.warning(f"Failed to extract item number for box {i}: {e}")
                item_number = "UNKNOWN"

            # Create/Update StudentAnswer for the first box automatically
            answer = session.exec(
                select(StudentAnswer).where(
                    StudentAnswer.paper_id == paper.paper_id,
                    StudentAnswer.item_id == item_id
                )
            ).first()
            if not answer:
                answer = StudentAnswer(
                    paper_id=paper.paper_id,
                    item_id=item_id,
                    image_directory=image_dir,
                    ai_evaluation="",
                    is_done_rendering=False,
                    detected_item_number=item_number
                )
                session.add(answer)
            else:
                answer.image_directory = image_dir
                answer.is_done_rendering = False
                answer.detected_item_number = item_number
            session.commit()
            session.refresh(answer)

        boxes_info.append({
            "index": i,
            "image_directory": image_dir,
            "item_number": item_number
        })
        
    return {
        "image_directory": default_image_dir,
        "num_boxes": len(processed_list),
        "boxes": boxes_info
    }

@app.patch("/api/test_instances/{test_id}/{student_no}/{item_id}")
async def update_answer_segmentation(
            test_id: str,
            student_no: str,
            item_id: int,
            file: UploadFile = File(...),
            points: str = Form(...),
            session: Session = Depends(get_session)
            ):
    """Update segmentation of student answer with manual points"""
    # ... [validation code remains unchanged] ...
    
    # Validate and parse points JSON (CRITICAL FIX BELOW)
    try:
        points_data = json.loads(points)
        required = ["ul", "ur", "lr", "ll"]
        for corner in required:
            # FIXED: Changed points_ → points_data (was causing NameError)
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
    
    # ===== IMAGE PROCESSING =====
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image content")
        
        src_pts = np.array([
            [float(points_data["ul"]["x"]), float(points_data["ul"]["y"])],
            [float(points_data["ur"]["x"]), float(points_data["ur"]["y"])],
            [float(points_data["lr"]["x"]), float(points_data["lr"]["y"])],
            [float(points_data["ll"]["x"]), float(points_data["ll"]["y"])],
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
        
        dst_pts = np.array([
            [0.0, 0.0],
            [float(OUT_WIDTH), 0.0],
            [float(OUT_WIDTH), float(OUT_HEIGHT)],
            [0.0, float(OUT_HEIGHT)]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (OUT_WIDTH, OUT_HEIGHT))
        
        preprocessor = CVImagePreprocessor(
            use_paddle_ocr=USE_PADDLE_OCR,
            paddle_ocr_lang=PADDLE_OCR_LANG,
            debug_mode=False  # Set to True for debugging during development
        )
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
    
    # Create or update TestPaperInstance
    paper = session.exec(
        select(TestPaperInstance).where(
            TestPaperInstance.test_id == test_id,
            TestPaperInstance.student_no == student_no
        )
    ).first()
    
    if not paper:
        paper = TestPaperInstance(
                test_id=test_id,
                student_no=student_no,
                is_done_rendering=False
                )
        session.add(paper)
        session.commit()
        session.refresh(paper)

    assert paper is not None, "Paper must exist after creation/lookup"
    
    # Create or update StudentAnswer
    answer = session.exec(
            select(StudentAnswer).where(
                StudentAnswer.paper_id == paper.paper_id,
                StudentAnswer.item_id == item_id
            )
            ).first()
    
    if not answer:
        answer = StudentAnswer(
                    paper_id=paper.paper_id,
                    item_id=item_id, 
                    image_directory=f"/api/temp/{safe_filename}",
                    ai_evaluation="",
                    is_done_rendering=False,
                    )
        session.add(answer)
        session.commit()
        session.refresh(answer)
    else:
        answer.image_directory = f"/api/temp/{safe_filename}"
        session.add(answer)

    session.commit()

    return {"image_directory": f"/api/temp/{safe_filename}"}


@app.get("/api/test_instances/{test_id}/statuses")
def get_test_paper_statuses(test_id: str, session: Session = Depends(get_session)):
    """Return per-student rendering status for a test instance"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Get all students in the section
    students = session.exec(
            select(Student).where(Student.section_id == instance.section_id)
            ).all()
    
    # Get all items for this test
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    
    # Build status response
    statuses = []
    for student in students:
        # Check if ALL items have processed answers
        all_items_processed = True
        for item in items:
            answer = session.exec(
                    select(StudentAnswer)
                        .join(TestPaperInstance)
                        .where(
                            TestPaperInstance.test_id == test_id,
                            TestPaperInstance.student_no == student.student_no,
                            StudentAnswer.item_id == item.item_id
                        )
                    ).first()
            
            if not answer or not answer.is_done_rendering:
                all_items_processed = False
                break
        
        statuses.append({
                "student_no": student.student_no,
                "name": student.name,
                "is_done_rendering": all_items_processed
                })
    
    return {
            "test_id": test_id,
            "statuses": statuses
            }


@app.get("/api/test_instances/{test_id}/results")
def get_ai_evaluation_results(test_id: str, session: Session = Depends(get_session)):
    """Return AI evaluations per contract"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    students = session.exec(
            select(Student).where(Student.section_id == instance.section_id)
            ).all()

    student_stores = []
    for student in students:
        papers = session.exec(
                    select(TestPaperInstance).where(
                        TestPaperInstance.test_id == test_id,
                        TestPaperInstance.student_no == student.student_no
                        )
                    ).all()
        
        ai_evaluations = []
        for paper in papers:
            answers = session.exec(
                        select(StudentAnswer).where(StudentAnswer.paper_id == paper.paper_id)
                        ).all()
            
            for answer in answers:
                respectiveItem = session.exec(
                                    select(TestItem).where(TestItem.item_id == answer.item_id)
                                    ).first()
                assert isinstance(respectiveItem, TestItem)

                # NOTE: commented out while API rate limit situation is unresolved.
                #===================EVALUATION CALLS=================
                # if not answer.is_done_rendering:
                #     print(f"INTERNAL:\tAttribute is_done_rendering is false for {answer.answer_id}.")
                #     _evaluate_image_logic(answer.answer_id, session)
                #     session.refresh(answer)
                #===================EVALUATION CALLS=================

                ai_evaluations.append({
                            "item_id": answer.item_id,
                            "answer_id": answer.answer_id,
                            "label": respectiveItem.label,
                            "question": respectiveItem.question,
                            "expected_answer_rubric_questions": respectiveItem.expected_answer_rubric_questions,
                            "ai_evaluation": answer.ai_evaluation if answer.ai_evaluation else ""
                            })
        
        student_stores.append({
                "student_no": student.student_no,
                "name": student.name,
                "evaluations": ai_evaluations
                })
    
    return {
            "test_id": test_id,
            "students": student_stores
            }


@app.get("/api/test_instances/{test_id}/results/{student_no}")
def get_ai_evaluation_results_per_student(
                test_id: str,
                student_no: str,
                session: Session = Depends(get_session)):
    """
    Get AI evaluation results for a specific student in a test instance.
    
    Returns AI grading results for a single student's submissions.
    Constructed by Jose.
    """
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    # Verify student exists
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
                )
    
    # Verify student belongs to the test's section
    if student.section_id != instance.section_id:
        raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Student '{student_no}' does not belong to section {instance.section_id}"
                )

    # Get test paper for this student
    paper = session.exec(
                select(TestPaperInstance)
                .where(
                    TestPaperInstance.test_id == test_id,
                    TestPaperInstance.student_no == student_no
                    )
                ).first()
    
    # Collect all AI evaluations for this student
    ai_evaluations = []
    if paper:
        answers = session.exec(
                        select(StudentAnswer)
                        .where(StudentAnswer.paper_id == paper.paper_id)
                        ).all()
        
        for answer in answers:
            respectiveItem = session.exec(
                                    select(TestItem).where(TestItem.item_id == answer.item_id)
                                    ).first()
            assert isinstance(respectiveItem, TestItem)

            # NOTE: commented out while API rate limit situation is unresolved.
            #===================EVALUATION CALLS=================
            # if answer.is_done_rendering == False:
            #     print(f"INTERNAL:\tAttribute is_done_rendering is false for {answer.answer_id}.")
            #     _evaluate_image_logic(answer.answer_id, session)
            #     session.refresh(answer)
            #===================EVALUATION CALLS=================
            
            ai_evaluations.append({
                        "item_id": answer.item_id,
                        "answer_id": answer.answer_id,
                        "label": respectiveItem.label,
                        "question": respectiveItem.question,
                        "expected_answer_rubric_questions": respectiveItem.expected_answer_rubric_questions,
                        "ai_evaluation": answer.ai_evaluation if answer.ai_evaluation else ""
                        })
    
    return {
        "test_id": test_id,
        "student_no": student_no,
        "name": student.name,
        "evaluations": ai_evaluations
        }


@app.get("/api/test_instances/{test_id}/{student_no}", response_model=List[StudentAnswerSummary])
def get_student_answers(
            test_id: str,
            student_no: str,
            session: Session = Depends(get_session),
            ):
    """Get all answers by a specific student for a test instance"""
    # Verify entities exist
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
                )
    
    # Get test paper
    paper = session.exec(
            select(TestPaperInstance).where(
                TestPaperInstance.test_id == test_id,
                TestPaperInstance.student_no == student_no
            )
            ).first()
    
    if not paper:
        return []  # No answers yet
    
    # Get all answers for this paper
    answers = session.exec(
        select(StudentAnswer)
        .where(StudentAnswer.paper_id == paper.paper_id)
        .order_by(StudentAnswer.item_id.asc()) # type: ignore[union-attr]
    ).all()
    
    # Build response with item labels
    summaries = []
    for answer in answers:
        item = session.get(TestItem, answer.item_id)
        if item:
            summaries.append(
                    StudentAnswerSummary(
                        student_no=student.student_no,
                        name=student.name,
                        item_id=answer.item_id,
                        label=item.label,
                        image_directory=answer.image_directory,
                        ai_evaluation=answer.ai_evaluation,
                        is_done_rendering=answer.is_done_rendering
                    ))
    
    return summaries
#endregion



# ==============================
#region Utility Endpoints
# ==============================
@app.post("/api/image_preprocess")
async def image_preprocess(file: UploadFile = File(...)):
    """Process raw student assessment image through CV pipeline (standalone)"""
    # Validation
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
    
    # Processing
    try:
        preprocessor = CVImagePreprocessor(
            use_paddle_ocr=USE_PADDLE_OCR,
            paddle_ocr_lang=PADDLE_OCR_LANG,
            debug_mode=False  # Set to True for debugging during development
        )
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
    
    # Save to temp files
    session_id = str(uuid.uuid4())
    boxes_info = []
    
    for i, img_bytes in enumerate(processed_list):
        filename = f"{session_id}_{i}.jpg"
        filepath = TEMP_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        
        boxes_info.append({
                "index": i,
                "image_directory": f"/api/temp/{filename}"
                })
    
    return {
            "num_boxes": len(processed_list),
            "boxes": boxes_info
            }


@app.get("/api/temp/{filename}")
async def get_processed_image(filename: str):
    """Serve processed images from temp directory"""
    # Security validation
    if not filename.endswith(".jpg") or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Allow only alphanumeric + safe characters
    if not all(c.isalnum() or c in "._-" for c in filename):
        raise HTTPException(status_code=400, detail="Invalid filename characters")
    
    filepath = TEMP_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return FileResponse(filepath, media_type="image/jpeg")


@app.patch("/api/answers/{answer_id}/reevaluate")
async def reevaluate_answer(answer_id: int, session: Session = Depends(get_session)):
    """Re-evaluate image then store to StudentAnswer evaluation result."""
    # TODO: 400 handling
    # TODO: 404 handling
    return _evaluate_image_logic(
                answer_id_input=answer_id,
                session=session
                )

#endregion



# ==============================
#   ---> FROM JOSE
#region Auxiliary Functions
# ==============================
def _evaluate_image_logic(answer_id_input: int, session: Session):
    _STRIP_POINTS = lambda x : re.sub(r'\s*\([^)]*\)\s*$', '', x).strip()
    _VALID_R_Q_RESPONSE = lambda x : x in ["YES", "NO"]
    _VALID_E_A_RESPONSE = lambda x : x in ["YES", "NO", "UNCLEAR"]

    print(f"INTERNAL:\tFunction evaluate_image({answer_id_input}) is being executed...")

    answer = session.exec(
                    select(StudentAnswer)
                    .where(StudentAnswer.answer_id == answer_id_input)
                    ).first()
    assert answer is not None

    actual_image_path = f"temp_cv_output/{answer.image_directory.split("/")[3]}"
    image_bytes: bytes = CV_IMAGE_PREPROCESSOR.load_image(actual_image_path)

    test_item = session.exec(
                    select(TestItem)
                    .where(TestItem.item_id == answer.item_id)
                    ).first()
    assert test_item is not None

    match test_item.is_problem_solving:
        case True:
            rubric_questions = test_item.expected_answer_rubric_questions.split(";")
            ai_evaluation = ""
            for rubric in rubric_questions:
                if rubric.strip() != "":
                    while True:
                        response = AI_ANSWER_EVALUATOR.evaluate_rubric(image_bytes, test_item.question, _STRIP_POINTS(rubric))
                        if response and _VALID_R_Q_RESPONSE(response):
                            break
                    ai_evaluation += f"{response};"
            
            answer.ai_evaluation = ai_evaluation
            # TODO: add missing setter for scores = list[float]
    
        case _:
            expected_answer = test_item.expected_answer_rubric_questions
            while True:
                response = AI_ANSWER_EVALUATOR.evaluate_expected_answer(image_bytes, test_item.question, _STRIP_POINTS(expected_answer))
                if response and _VALID_E_A_RESPONSE(response):
                    break

            answer.ai_evaluation = response
            # TODO: add missing setter for scores = list[float]

    answer.is_done_rendering = True

    session.add(answer)
    session.commit()
    session.refresh(answer)

    return StudentAnswerResponse(
            answer_id=answer.answer_id,
            paper_id=answer.paper_id,
            item_id=answer.item_id,
            image_directory=answer.image_directory,
            ai_evaluation=answer.ai_evaluation,
            is_done_rendering=answer.is_done_rendering
            # TODO: add missing scores = list[float]
            )


def _populate_spreadsheet_logic(test_id_input: str, session: Session):
    test_items = session.exec(
                            select(TestItem)
                            .where(TestItem.test_id == test_id_input)
                            ).all()
    
    test_items_labels = [item.label for item in test_items]

    SHEET = SheetsExporter(columns=test_items_labels)

    students = session.exec(select(Student).where())
    # TODO: complete the rest of the function
    ...

#endregion
