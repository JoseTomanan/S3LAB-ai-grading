import base64
from multiprocessing import process
from fastapi import APIRouter, HTTPException, Response, status, Depends, File, UploadFile, Form, Query
from sqlmodel import Session, select
from typing import List, Optional

import uuid
import json
from pathlib import Path

from models import *
from schemas import *
from database import get_session

from services.box_segmenter import BoxSegmenter
from services.document_scanner import DocumentScanner

from services.utility import *


router = APIRouter()
TEMP_DIR = Path("static/images")


# ==============================
#region Endpoints
@router.post("/{test_id}/{student_no}/image_preprocess")
async def process_student_answer_image(
                test_id: str,
                student_no: str,
                file: UploadFile = File(...),
                num_boxes: Optional[int] = Query(None), 
                session: Session = Depends(get_session)
                ):
    """Process raw student assessment image through CV pipeline"""
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

    print(f"INTERNAL:\tValidation checks have passed. Processing and segmenting now.")

    # ======== DOCUMENT SCANNING ========
    DOCUMENT_SCANNER = DocumentScanner()
    scanned_page = DOCUMENT_SCANNER.scan_page(contents)

    # ======== BOX SEGMENTING ========
    BOX_SEGMENTER = BoxSegmenter()
    segmented_list: list[bytes] = BOX_SEGMENTER.get_boxes(scanned_page, num_boxes if num_boxes is not None else 3)
    processed_list: list[bytes] = [BOX_SEGMENTER.beautify_scan(b) for b in segmented_list]
    
    print(f"INTERNAL:\tSegmenting success with {len(processed_list)} boxes detected.")

    # ===== SAVE ALL CANDIDATE BOXES FOR PREVIEW =====
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

    print(f"INTERNAL:\tProceeding to labeling the boxes.")
    boxes_info = _label_save_commit_boxes(test_id, student_no, paper, processed_list, session)

    return {
        "num_boxes": len(processed_list),
        "boxes": boxes_info,
        }


@router.patch("/{test_id}/{student_no}/{item_id}")
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
    BOX_SEGMENTER = BoxSegmenter()
    img_bytes_crop = crop_image(contents, points_data)
    img_bytes = BOX_SEGMENTER.beautify_scan(img_bytes_crop)
    
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


@router.get("/{test_id}/statuses")
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


@router.get("/{test_id}/results")
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


@router.get("/{test_id}/results/{student_no}")
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


@router.get("/{test_id}/{student_no}", response_model=List[StudentAnswerSummary])
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
#region Auxiliary functions    
def _label_save_commit_boxes(
                test_id: str,
                student_no: str,
                paper: TestPaperInstance,
                processed_list: list[bytes],
                session: Session
                ):
    boxes_info_step1 = _label_save_boxes(test_id, student_no, paper, processed_list, session)
    boxes_info = _commit_boxes(boxes_info_step1, paper, session)
    return boxes_info


def _label_save_boxes(
                test_id: str,
                student_no: str,
                paper: TestPaperInstance,
                processed_list: list[bytes],
                session: Session
                ):
    boxes_info = []
    for i, img_bytes in enumerate[bytes](processed_list):
        # Extract item number using AI
        test_item_labels = [ti.label
                    for ti in session.exec(
                        select(TestItem).where(
                            TestItem.test_id == test_id
                    )) if ti.label is not None]

        try:
            item_number = AI_ANSWER_EVALUATOR.get_nearest_item_number(img_bytes, test_item_labels)
            if not item_number or item_number.strip() == "NONE":
                item_number = "NONE"
            else:
                item_number = item_number.strip()
        except Exception as e:
            print(f"INTERNAL:\tFailed to extract item number for box {i}: {e}")
            item_number = "NONE"

        print(f"INTERNAL:\t{i}th detected label = {item_number}")

        test_item = session.exec(
                        select(TestItem).where(
                            TestItem.label == item_number,
                        )).first()
        
        if test_item is None:
            raise Exception("This is not supposed to happen. Read your code again.")

        item_id = test_item.item_id
        print(f"INTERNAL:\tLabel {item_number} will be stored in {item_id}")

        # Generate filename
        safe_filename = f"{test_id}_{student_no}_{uuid.uuid4().hex}_{i}.jpg"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        filepath = TEMP_DIR / safe_filename
        
        # Save image file
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        
        image_dir = f"/api/temp/{safe_filename}"

        boxes_info.append({
                    "index": i,
                    "image_directory": image_dir,
                    "item_number": item_number
                    })

    return boxes_info


def _commit_boxes(
                boxes_info: list[dict], # TODO: add type safety
                paper: TestPaperInstance,
                session: Session
                ):
    for box in boxes_info:
        item_id = box.test_item.item_id
        image_dir = box.image_directory
        item_number = box.item_number
        
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
            answer.ai_evaluation=""
            answer.is_done_rendering = False
            answer.detected_item_number = item_number
        session.commit()
        session.refresh(answer)
#endregion
# ==============================
