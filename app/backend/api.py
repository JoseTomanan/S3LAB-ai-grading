from fastapi import FastAPI, HTTPException, Response, status, Depends, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, delete, select
from typing import List, Optional

import uuid
import json
from pathlib import Path

from models import *
from schemas import *
from database import create_db_and_tables, get_session

from functions.ai_interface import AIAnswerEvaluator
from functions.box_segmenter import BoxSegmenter
from functions.document_scanner import DocumentScanner
from functions.image_modifier import ImageModifier

from routes import sections, students, utility, test_instances

from services import *



# ==============================
#region App Initialization
app = FastAPI(
        title="Assessment Processing API",
        description="API for managing test instances, items, and student answer processing",
        lifespan=create_db_and_tables(),
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
DOCUMENT_SCANNER = DocumentScanner()
IMAGE_MODIFIER = ImageModifier()

TEMP_DIR = Path("static/images")
TEMP_DIR.mkdir(exist_ok=True)

#endregion
# ==============================


app.include_router(sections.router, prefix="/api/sections", tags=["Sections"])
app.include_router(students.router, prefix="/api/students", tags=["Students"])
app.include_router(utility.router, prefix="/api", tags=["Utility"])
app.include_router(test_instances.router, prefix="/api/test_instances", tags=["Test Instances"])


# ==============================
#region Test Item Endpoints
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



# ==============================
#region Student Answer Processing Endpoints
@app.post("/api/test_instances/{test_id}/{student_no}/image_preprocess")
async def process_student_answer_image(
                test_id: str,
                student_no: str,
                file: UploadFile = File(...),
                num_boxes: Optional[int] = Query(None), 
                session: Session = Depends(get_session)
                ):
    """Process raw student assessment image through CV pipeline"""
    print(f"INTERNAL:\tProcess student answer image request has been received.")

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
    
    # Validate file
    if not file or not file.filename:
        raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No file provided"
                    )
    if not IMAGE_MODIFIER.validate_file_extension(file.filename):
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
    print(f"INTERNAL:\tValidation checks have passed. Processing and segmenting now.")
    BOX_SEGMENTER = BoxSegmenter()

    scanned_page = BOX_SEGMENTER.scan_page(contents)
    processed_list = BOX_SEGMENTER.get_boxes(
                            scanned_page,
                            num_boxes=num_boxes if num_boxes is not None else 3
                            )
    
    print(f"INTERNAL:\tSegmenting success with {len(processed_list)} boxes detected.")

    # ===== SAVE ALL CANDIDATE BOXES FOR PREVIEW =====
    # Create or update TestPaperInstance (metadata only, no specific answer linked yet)
    paper = session.exec(
                select(TestPaperInstance).where(
                    TestPaperInstance.test_id == test_id,
                    TestPaperInstance.student_no == student_no
                )).first()
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

    print(f"INTERNAL:\tProceeding to labeling the boxes.")
    for i, img_bytes in enumerate(processed_list):
        # Generate filename
        safe_filename = f"{test_id}_{student_no}_{uuid.uuid4().hex}_{i}.jpg"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        filepath = TEMP_DIR / safe_filename
        
        # Save image file
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        
        image_dir = f"/api/temp/{safe_filename}"

        # Extract item number using AI
        item_number = "UNKNOWN"
        try:
            item_number = AI_ANSWER_EVALUATOR.get_item_number(img_bytes)
            if not item_number or item_number.strip() == "UNKNOWN":
                item_number = "UNKNOWN"
            else:
                item_number = item_number.strip()
        except Exception as e:
            print(f"INTERNAL:\tFailed to extract item number for box {i}: {e}")
            item_number = "UNKNOWN"

        print(f"INTERNAL:\t{i}th detected label = {item_number}")

        test_item = session.exec(
                        select(TestItem).where(
                            TestItem.label == item_number,
                        )).first()
        
        if test_item is None:
            # TODO: fix exception handling
            raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Sorry Ma'am di pa tapos",
                        )

        item_id = test_item.item_id
        print(f"INTERNAL:\tLabel {item_number} will be stored in {item_id}")

        answer = session.exec(
                    select(StudentAnswer).where(
                        StudentAnswer.paper_id == paper.paper_id,
                        StudentAnswer.item_id == item_id
                    )).first()
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
    contents = await file.read()
    img_bytes = crop_image(contents, points_data)
    
    # ===== STORAGE & RESPONSE =====
    safe_filename = f"{test_id}_{student_no}_{item_id}_{uuid.uuid4().hex}.jpg"
    filepath = TEMP_DIR / safe_filename
    
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    
    # Create or update TestPaperInstance
    paper = session.exec(
                select(TestPaperInstance).where(
                    TestPaperInstance.test_id == test_id,
                    TestPaperInstance.student_no == student_no
                )).first()
    
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
        answer.ai_evaluation = ""
        answer.is_done_rendering = False
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
        total_score = 0
        max_score = 0

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

            else:
                _parsed_scores = get_total_score(item.expected_answer_rubric_questions, answer.ai_evaluation)
                total_score += _parsed_scores[0]
                max_score += _parsed_scores[1]

        statuses.append({
                "student_no": student.student_no,
                "name": student.name,
                "is_done_rendering": all_items_processed,
                "total_score": f"{total_score}/{max_score}",
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
            
            scores = calculate_score(
                                respectiveItem.expected_answer_rubric_questions,
                                answer.ai_evaluation
                                )
            
            ai_evaluations.append({
                        "item_id": answer.item_id,
                        "answer_id": answer.answer_id,
                        "label": respectiveItem.label,
                        "question": respectiveItem.question,
                        "expected_answer_rubric_questions": respectiveItem.expected_answer_rubric_questions,
                        "ai_evaluation": answer.ai_evaluation if answer.ai_evaluation else "",
                        "scores": scores,
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
