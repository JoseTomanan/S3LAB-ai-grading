from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, status, Depends, File, UploadFile, Form, Query
from sqlmodel import Session, select
from typing import List, Optional
import uuid
import json
from pathlib import Path

from models import *
from schemas import *
from core.database import get_session, get_direct_session
from logic.box_segmenter import BoxSegmenter
from logic.document_scanner import DocumentScanner
from logic.utility import *



router = APIRouter()
TEMP_DIR = Path("static/images")


# ==============================
#region Endpoints
@router.post("/{test_id}/{student_no}/image_preprocess")
async def process_student_answer_image(
                test_id: str,
                student_no: str,
                background_tasks: BackgroundTasks,
                files: List[UploadFile] = File(...),
                num_boxes: Optional[int] = Query(None),
                session: Session = Depends(get_session)
                ):
    """Process raw student assessment image(s) through CV pipeline. Accepts multiple pages."""
    await _validate_request(test_id, student_no, session)
    contents_list = await _validate_files(files)

    print(f"INFO:\tValidation checks have passed. Processing and segmenting {len(contents_list)} page(s) now.")
    processed_list: list[bytes] = await _scan_and_segment_pages(contents_list, num_boxes)

    # ===== SAVE ALL CANDIDATE BOXES FOR PREVIEW =====
    print(f"INFO:\tProceeding to labeling the boxes.")
    boxes_info = _label_save_boxes(test_id, student_no, processed_list, session)
    answer_ids = _create_answer_records(boxes_info, test_id, student_no, session)
    background_tasks.add_task(_evaluate_answers_background, answer_ids)

    return Response(
            status_code=status.HTTP_202_ACCEPTED,
            content=json.dumps({
                "num_boxes": len(processed_list),
                "boxes": boxes_info,
            }),
            media_type="application/json"
            )


@router.post("/{test_id}/{student_no}/label_save_boxes")
async def scan_then_label_save_boxes(
                test_id: str,
                student_no: str,
                files: List[UploadFile] = File(...),
                num_boxes: Optional[int] = Query(None),
                session: Session = Depends(get_session)
                ):
    """Process raw student assessment image(s) through CV pipeline. Accepts multiple pages."""
    await _validate_request(test_id, student_no, session)
    contents_list = await _validate_files(files)

    print(f"INFO:\tValidation checks have passed. Processing and segmenting {len(contents_list)} page(s) now.")
    try:
        processed_list: list[bytes] = await _scan_and_segment_pages(contents_list, num_boxes)
    except:
        raise HTTPException(status_code=500, detail="Could not find any boxes.")

    print(f"INFO:\tProceeding to labeling the boxes.")
    boxes_info = _label_save_boxes(test_id, student_no, processed_list, session)

    return {
            "num_boxes": len(processed_list),
            "boxes": boxes_info,
            }


@router.post("/{test_id}/{student_no}/commit_boxes")
async def commit_boxes_endpoint(
                test_id: str,
                student_no: str,
                request_body: CommitBoxesRequest,
                background_tasks: BackgroundTasks,
                session: Session = Depends(get_session)
                ):
    test_exists = session.exec(
                    select(TestInstance)
                    .where(TestInstance.test_id == test_id)
                    ).first()
    if not test_exists:
        print("INFO:\ttest_exists failed")
        raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Test with ID '{test_id}' not found"
                    )

    student_exists = session.exec(
                        select(Student)
                        .where(Student.student_no == student_no)
                        ).first()
    if not student_exists:
        print("INFO:\tstudent_exists failed")
        raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Student with student_no '{student_no}' not found"
                    )

    boxes_info = [box.model_dump() for box in request_body.boxes]
    answer_ids = _create_answer_records(boxes_info, test_id, student_no, session)
    background_tasks.add_task(_evaluate_answers_background, answer_ids)

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.patch("/{test_id}/{student_no}/{item_id}")
async def update_answer_segmentation(
            test_id: str,
            student_no: str,
            item_id: int,
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            points: str = Form(...),
            session: Session = Depends(get_session)
            ):
    """Update segmentation of student answer with manual points"""
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

    # ===== RESOLVE ITEM LABEL =====
    test_item = session.get(TestItem, item_id)
    if not test_item:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID '{item_id}' not found"
                )

    # ===== IMAGE PROCESSING =====
    contents = await file.read()
    BOX_SEGMENTER = BoxSegmenter()
    img_bytes_crop = crop_image(contents, points_data)
    img_bytes = BOX_SEGMENTER.beautify_scan(img_bytes_crop)

    # ===== SAVE IMAGE =====
    safe_filename = f"{test_id}_{student_no}_{item_id}_{uuid.uuid4().hex}.jpg"
    filepath = TEMP_DIR / safe_filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    image_dir = f"/api/temp/{safe_filename}"

    # ===== COMMIT RECORDS SYNCHRONOUSLY, EVALUATE IN BACKGROUND =====
    boxes_info = [{"image_directory": image_dir, "item_number": test_item.label}]
    answer_ids = _create_answer_records(boxes_info, test_id, student_no, session)
    background_tasks.add_task(_evaluate_answers_background, answer_ids)

    return Response(
            status_code=status.HTTP_202_ACCEPTED,
            content=json.dumps({"image_directory": image_dir}),
            media_type="application/json"
            )


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
        has_any_answer = False
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

            if not answer:
                all_items_processed = False
                continue

            has_any_answer = True
            if not answer.is_done_rendering:
                all_items_processed = False
            else:
                _parsed_scores = get_total_score(item.expected_answer_rubric_questions, answer.ai_evaluation)
                total_score += _parsed_scores[0]
                max_score += _parsed_scores[1]

        statuses.append({
                "student_no": student.student_no,
                "name": student.name,
                "is_done_rendering": all_items_processed or not has_any_answer,
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


@router.delete("/{item_id}/{student_no}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student_answer(
                item_id: int,
                student_no: str,
                session: Session = Depends(get_session),
                ):
    """Delete a single student answer and its associated image file, if any."""
    # Find the TestPaperInstance for this student and test paper (by student_no)
    paper = session.exec(
                select(TestPaperInstance)
                .where(TestPaperInstance.student_no == student_no)
                ).first()
    if not paper:
        raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Test paper for student_no '{student_no}' not found"
                    )

    # Now find the StudentAnswer by (item_id, paper_id)
    answer = session.exec(
                select(StudentAnswer)
                .where(StudentAnswer.item_id == item_id, StudentAnswer.paper_id == paper.paper_id)
                ).first()
    if not answer:
        raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Student answer with item_id '{item_id}' and student_no '{student_no}' not found"
                    )

    # Best-effort cleanup of the stored image file
    image_path = None
    if answer.image_directory:
        # Stored as something like "/api/temp/<filename>"
        filename = answer.image_directory.split("/")[-1]
        if filename:
            image_path = TEMP_DIR / filename

    session.delete(answer)
    session.commit()

    if image_path is not None and image_path.exists():
        try:
            image_path.unlink()
        except Exception:
            # Do not fail the API call if file deletion fails
            pass

    return Response(status_code=status.HTTP_204_NO_CONTENT)
#endregion



# ==============================
#region Auxiliary functions    
# TODO: Relocate to backend/logic/student_answers_utils.py (or a similar module)
# according to FastAPI best practices. Utility/data-layer code should not reside in router files.
async def _validate_request(test_id: str, student_no: str, session: Session):
    if not session.get(TestInstance, test_id):
        raise HTTPException(status_code=404, detail=f"Test instance '{test_id}' not found")
    if not session.get(Student, student_no):
        raise HTTPException(status_code=404, detail=f"Student with ID '{student_no}' not found")


async def _validate_files(files: List[UploadFile]) -> list[bytes]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    contents_list = []
    for file in files:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        if not IMAGE_MODIFIER.validate_file_extension(file.filename):
            raise HTTPException(status_code=415, detail=f"Unsupported file format. Got: {file.filename}")
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        contents_list.append(contents)
    return contents_list


async def _scan_and_segment_pages(contents_list: list[bytes], num_boxes: Optional[int]) -> list[bytes]:
    DOCUMENT_SCANNER = DocumentScanner()
    BOX_SEGMENTER = BoxSegmenter()

    all_processed: list[bytes] = []
    for page_idx, contents in enumerate(contents_list):
        # ======== DOCUMENT SCANNING ========
        scanned_page = DOCUMENT_SCANNER.scan_page(contents)

        # ======== BOX SEGMENTING ========
        segmented_list: list[bytes] = BOX_SEGMENTER.get_answer_sections(scanned_page, num_boxes if num_boxes is not None else 3)
        processed_list: list[bytes] = [BOX_SEGMENTER.beautify_scan(b) for b in segmented_list]

        print(f"INFO:\tPage {page_idx + 1}: segmented {len(processed_list)} boxes.")
        all_processed.extend(processed_list)

    if num_boxes is not None:
        all_processed = all_processed[:num_boxes]

    print(f"INFO:\tTotal boxes across {len(contents_list)} page(s): {len(all_processed)}")
    return all_processed


def _label_save_boxes(
                test_id: str,
                student_no: str,
                processed_list: list[bytes],
                session: Session
                ) -> list[dict]:
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
            print(f"INFO:\tFailed to extract item number for box {i}: {e}")
            item_number = "NONE"

        print(f"INFO:\t{i}th detected label = {item_number}")

        test_item = session.exec(
                        select(TestItem).where(
                            TestItem.label == item_number,
                        )).first()
        
        if test_item is None:
            print(f"INFO:\tItem not found because label={item_number} is not valid. Continuing...")
            continue

        item_id = test_item.item_id
        print(f"INFO:\tLabel {item_number} will be stored in {item_id}")

        # Generate filename
        safe_filename = f"{test_id}_{student_no}_{uuid.uuid4().hex[:6]}_{i}.jpg"
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


def _create_answer_records(
                boxes_info: list[dict],
                test_id: str,
                student_no: str,
                session: Session
                ) -> list[int]:
    """Create TestPaperInstance and StudentAnswer records synchronously. Returns answer_ids."""
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

    answer_ids = []
    for box in boxes_info:
        image_dir = box["image_directory"]
        item_number = box["item_number"]
        item = session.exec(
                        select(TestItem).where(
                            TestItem.label == item_number,
                        )).first()
        assert item is not None
        item_id = item.item_id

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
            answer.ai_evaluation = ""
            answer.is_done_rendering = False
            answer.detected_item_number = item_number
        session.commit()
        session.refresh(answer)
        answer_ids.append(answer.answer_id)

    return answer_ids


def _evaluate_answers_background(answer_ids: list[int]):
    """Run AI evaluation on pre-created answer records in a background task."""
    session = get_direct_session()
    try:
        for answer_id in answer_ids:
            try:
                evaluate_image_logic(answer_id, session)
            except Exception as e:
                print(f"INFO:\tAI evaluation failed for answer {answer_id}: {e}")
                answer = session.get(StudentAnswer, answer_id)
                if answer:
                    answer.is_done_rendering = True
                    answer.ai_evaluation = f"_ERROR: {e}"
                    session.commit()
    except Exception as e:
        print(f"INFO:\tBackground evaluation failed: {e}")
        session.rollback()
    finally:
        session.close()
#endregion
# ==============================
