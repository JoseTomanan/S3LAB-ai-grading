from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, status, Depends, File, UploadFile, Form, Query
from sqlmodel import Session, select
from typing import List, Optional
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
@router.post("/{test_id}/{student_no}/image_preprocess", response_model=ImagePreprocessResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_student_answer_image(
                test_id: str,
                student_no: str,
                background_tasks: BackgroundTasks,
                files: List[UploadFile] = File(...),
                num_boxes: Optional[int] = Query(None),
                is_scanned_already: bool = Query(False),
                session: Session = Depends(get_session)
                ):
    """
    Process raw student assessment image(s) through CV pipeline. Accepts multiple pages.
    Works in one straight line, without validation --- to be used by Bulk Upload feature.
    """
    await _validate_request(test_id, student_no, session)
    contents_list = await _validate_files(files)

    print(f"BACKEND:\tValidation checks have passed. Processing and segmenting {len(contents_list)} page(s) now.")
    try:
        processed_list: list[bytes] = await _scan_and_segment_pages(contents_list, num_boxes, is_scanned_already)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ===== SAVE ALL CANDIDATE BOXES FOR PREVIEW =====
    print(f"BACKEND:\tProceeding to labeling the boxes.")
    boxes_info = _label_save_boxes(test_id, student_no, processed_list, session)
    answer_ids = _create_answer_records(boxes_info, test_id, student_no, session)
    background_tasks.add_task(_evaluate_answers_background, answer_ids)

    return ImagePreprocessResponse(
            num_boxes=len(processed_list),
            boxes=[BoxesItem(**b) for b in boxes_info],
            )


@router.post("/{test_id}/{student_no}/label_save_boxes", response_model=LabelSaveBoxesResponse)
async def scan_then_label_save_boxes(
                test_id: str,
                student_no: str,
                files: List[UploadFile] = File(...),
                num_boxes: Optional[int] = Query(None),
                is_scanned_already: bool = Query(False),
                session: Session = Depends(get_session)
                ):
    """
    Process raw student assessment image(s) through CV pipeline. Accepts multiple pages.
    Waits for validation before proceeding --- to be used by (individual) Process Image functionality.
    """
    await _validate_request(test_id, student_no, session)
    contents_list = await _validate_files(files)

    print(f"BACKEND:\tValidation checks have passed. Processing and segmenting {len(contents_list)} page(s) now.")
    try:
        processed_list: list[bytes] = await _scan_and_segment_pages(contents_list, num_boxes, is_scanned_already)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    print(f"BACKEND:\tProceeding to labeling the boxes.")
    boxes_info = _label_save_boxes(test_id, student_no, processed_list, session)

    return LabelSaveBoxesResponse(
            num_boxes=len(processed_list),
            boxes=[BoxesItem(**b) for b in boxes_info],
            )


@router.post("/{test_id}/{student_no}/commit_boxes")
async def commit_boxes_endpoint(
                test_id: str,
                student_no: str,
                request_body: CommitBoxesRequest,
                background_tasks: BackgroundTasks,
                session: Session = Depends(get_session)
                ):
    """Second endpoint called in conjunction with scan_then_label_save_boxes endpoint."""
    test_exists = session.exec(
                    select(TestInstance)
                    .where(TestInstance.test_id == test_id)
                    ).first()
    if not test_exists:
        print("BACKEND:\ttest_exists failed")
        raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Test with ID '{test_id}' not found"
                    )

    student_exists = session.exec(
                        select(Student)
                        .where(Student.student_no == student_no)
                        ).first()
    if not student_exists:
        print("BACKEND:\tstudent_exists failed")
        raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Student with student_no '{student_no}' not found"
                    )

    boxes_info = [box.model_dump() for box in request_body.boxes]
    boxes_info_bad = [b for b in boxes_info if b["index"] < 0]
    boxes_info_good = [b for b in boxes_info if b["index"] >= 0]

    _delete_disregarded_record_imgs(boxes_info_bad)

    answer_ids = _create_answer_records(boxes_info_good, test_id, student_no, session)
    background_tasks.add_task(_evaluate_answers_background, answer_ids)

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.patch("/{test_id}/{student_no}/{item_id}", response_model=UpdateSegmentationResponse, status_code=status.HTTP_202_ACCEPTED)
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
    safe_filename = f"{test_id}_{student_no}_{uuid.uuid4().hex[:6]}_{item_id}.jpg"
    filepath = TEMP_DIR / safe_filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    image_dir = f"/api/temp/{safe_filename}"

    # ===== COMMIT RECORDS SYNCHRONOUSLY, EVALUATE IN BACKGROUND =====
    boxes_info = [{"image_directory": image_dir, "item_number": test_item.label}]
    answer_ids = _create_answer_records(boxes_info, test_id, student_no, session)
    background_tasks.add_task(_evaluate_answers_background, answer_ids)

    return UpdateSegmentationResponse(image_directory=image_dir)


@router.get("/{test_id}/statuses", response_model=TestStatusesResponse)
def get_test_paper_statuses(test_id: str, session: Session = Depends(get_session)):
    """Return per-student rendering status for a test instance"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    ## Get all students in the section
    students = session.exec(
            select(Student).where(Student.section_id == instance.section_id)
            ).all()
    
    ## Get all items for this test
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    
    statuses = []
    for student in students:
        total_score = 0
        max_score = 0

        ## We use 'all_items_processed' to check if this student has valid (rendered) answers for every item in the test
        ## 'all_items_processed' starts as True and is set to False if any test item is missing a corresponding answer, 
        ## or if the AI evaluation of any answer is not finished (is_done_rendering == False).
        ## 'has_any_answer' tracks if student has >=1 answer for any item -- distinguish between "no answers yet" vs "all answers processed".
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
                ## No answer was found for this item (by this student); update state.
                all_items_processed = False
                continue

            ## There is at least one answer for this test+student.
            has_any_answer = True
            if not answer.is_done_rendering:
                ## This answer exists but is still being processed by the AI (not fully evaluated yet)
                all_items_processed = False
            else:
                ## This answer is done and has an AI rubric evaluation; aggregate scores.
                _parsed_scores = get_total_score(item.expected_answer_rubric_questions, answer.ai_evaluation)
                total_score += _parsed_scores[0]
                max_score += _parsed_scores[1]

        statuses.append(StudentStatusItem(
                student_no=student.student_no,
                name=student.name,
                is_done_rendering=all_items_processed and has_any_answer,
                has_any_answer=has_any_answer,
                total_score=f"{total_score}/{max_score}",
                ))

    return TestStatusesResponse(test_id=test_id, statuses=statuses)


@router.get("/{test_id}/results", response_model=TestResultsResponse)
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

                ai_evaluations.append(AIEvaluationItem(
                            item_id=answer.item_id,
                            answer_id=answer.answer_id,
                            label=respectiveItem.label,
                            question=respectiveItem.question,
                            expected_answer_rubric_questions=respectiveItem.expected_answer_rubric_questions,
                            ai_evaluation=answer.ai_evaluation if answer.ai_evaluation else "",
                            ))

        student_stores.append(StudentEvaluationsStore(
                student_no=student.student_no,
                name=student.name,
                evaluations=ai_evaluations,
                ))

    return TestResultsResponse(test_id=test_id, students=student_stores)


@router.get("/{test_id}/results/{student_no}", response_model=StudentResultsResponse)
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
            
            ai_evaluations.append(AIEvaluationItemWithScores(
                        item_id=answer.item_id,
                        answer_id=answer.answer_id,
                        label=respectiveItem.label,
                        question=respectiveItem.question,
                        expected_answer_rubric_questions=respectiveItem.expected_answer_rubric_questions,
                        ai_evaluation=answer.ai_evaluation if answer.ai_evaluation else "",
                        scores=scores,
                        ))

    return StudentResultsResponse(
        test_id=test_id,
        student_no=student_no,
        name=student.name,
        evaluations=ai_evaluations,
        )


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


async def _scan_and_segment_pages(contents_list: list[bytes], num_boxes: Optional[int], is_scanned_already: bool = False) -> list[bytes]:
    DOCUMENT_SCANNER = DocumentScanner()
    BOX_SEGMENTER = BoxSegmenter()

    all_processed: list[bytes] = []
    for page_idx, contents in enumerate(contents_list):
        ## ======== DOCUMENT SCANNING ========
        contents = DOCUMENT_SCANNER.normalize_bytes(contents)
        scanned_page = contents if is_scanned_already else DOCUMENT_SCANNER.scan_page(contents)

        ## ======== BOX SEGMENTING ========
        segmented_list: list[bytes] = BOX_SEGMENTER.get_answer_sections(scanned_page, num_boxes if num_boxes is not None else 3)
        processed_list: list[bytes] = [BOX_SEGMENTER.beautify_scan(b) for b in segmented_list]

        print(f"BACKEND:\tPage {page_idx + 1}: segmented {len(processed_list)} boxes.")
        all_processed.extend(processed_list)

    if num_boxes is not None:
        all_processed = all_processed[:num_boxes]

    print(f"BACKEND:\tTotal boxes across {len(contents_list)} page(s): {len(all_processed)}")
    return all_processed


def _label_save_boxes(
                test_id: str,
                student_no: str,
                processed_list: list[bytes],
                session: Session
                ) -> list[dict]:
    ## Phase A: Hoist DB query — run once instead of once per box
    test_item_labels = [ti.label
                for ti in session.exec(
                    select(TestItem).where(
                        TestItem.test_id == test_id
                )) if ti.label is not None]

    ## Phase B: Parallelize Gemini API calls
    def _classify_box(index: int, img_bytes: bytes) -> tuple[int, str]:
        try:
            item_number = AI_ANSWER_EVALUATOR.get_nearest_item_number(img_bytes, test_item_labels)
            if not item_number or item_number.strip() == "NONE":
                return (index, "NONE")
            return (index, item_number.strip())
        except Exception as e:
            print(f"BACKEND:\tFailed to extract item number for box {index}: {e}")
            return (index, "NONE")

    if not processed_list:
        return []

    ai_results: list[tuple[int, str] | None] = [None] * len(processed_list)
    with ThreadPoolExecutor(max_workers=min(len(processed_list), 10)) as executor:
        futures = {
            executor.submit(_classify_box, i, img_bytes): i
            for i, img_bytes in enumerate(processed_list)
        }
        for future in as_completed(futures):
            try:
                index, item_number = future.result()
                ai_results[index] = (index, item_number)
            except Exception as e:
                i = futures[future]
                print(f"BACKEND:\tUnexpected error classifying box {i}: {e}")
                ai_results[i] = (i, "NONE")

    ## Phase C: Sequential post-processing on main thread (DB lookups + file writes)
    boxes_info = []
    for result in ai_results:
        if result is None:
            continue
        index, item_number = result
        print(f"BACKEND:\t{index}th detected label = {item_number}")

        test_item = session.exec(
                        select(TestItem).where(
                            TestItem.test_id == test_id,
                            TestItem.label == item_number,
                        )).first()

        if test_item is None:
            print(f"BACKEND:\tItem not found because label={item_number} is not valid. Continuing...")
            continue

        print(f"BACKEND:\tLabel {item_number} will be stored in {test_item.item_id}")

        ## Generate filename
        img_bytes = processed_list[index]
        safe_filename = f"{test_id}_{student_no}_{uuid.uuid4().hex[:6]}_{index}.jpg"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        filepath = TEMP_DIR / safe_filename

        ## Save image file
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        image_dir = f"/api/temp/{safe_filename}"

        boxes_info.append({
                    "index": index,
                    "image_directory": image_dir,
                    "item_number": item_number
                    })

    return boxes_info


def _delete_disregarded_record_imgs(boxes_info: list[dict]):
    """Delete image files for boxes marked as discarded (index < 0)."""
    for box in boxes_info:
        image_dir = box.get("image_directory")
        if not image_dir:
            continue
        filename = image_dir.split("/")[-1]
        if filename:
            image_path = TEMP_DIR / filename
            if image_path.exists():
                try:
                    image_path.unlink()
                except Exception as e:
                    print(f"BACKEND:\tFailed to delete discarded image {filename}: {e}")


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
                            TestItem.test_id == test_id,
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
    if not answer_ids:
        return

    session = get_direct_session()
    try:
        ## Phase A: Sequential DB reads — load image bytes and test items for all answers
        eval_tasks: list[tuple[int, bytes, TestItem]] = []
        for answer_id in answer_ids:
            answer = session.get(StudentAnswer, answer_id)
            if answer is None:
                continue
            actual_image_path = f"static/images/{answer.image_directory.split('/')[3]}"
            image_bytes = DOCUMENT_SCANNER.load_image(actual_image_path)
            test_item = session.get(TestItem, answer.item_id)
            if test_item is None:
                continue
            eval_tasks.append((answer_id, image_bytes, test_item))

        ## Phase B: Parallel AI calls — no DB access inside threads
        def _run_ai(answer_id: int, image_bytes: bytes, test_item: TestItem) -> tuple[int, str | None, Exception | None]:
            try:
                ai_evaluation = compute_ai_evaluation(image_bytes, test_item)
                return (answer_id, ai_evaluation, None)
            except Exception as e:
                print(f"BACKEND:\tAI evaluation failed for answer {answer_id}: {e}")
                return (answer_id, None, e)

        ai_results: list[tuple[int, str | None, Exception | None] | None] = [None] * len(eval_tasks)
        with ThreadPoolExecutor(max_workers=min(len(eval_tasks), 10)) as executor:
            futures = {
                executor.submit(_run_ai, *task): i
                for i, task in enumerate(eval_tasks)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    ai_results[i] = future.result()
                except Exception as e:
                    answer_id = eval_tasks[i][0]
                    print(f"BACKEND:\tUnexpected error evaluating answer {answer_id}: {e}")
                    ai_results[i] = (answer_id, None, e)

        ## Phase C: Sequential DB writes
        for result in ai_results:
            if result is None:
                continue
            answer_id, ai_evaluation, error = result
            answer = session.get(StudentAnswer, answer_id)
            if answer is None:
                continue
            if error:
                answer.is_done_rendering = True
                answer.ai_evaluation = f"_ERROR: {error}"
            else:
                assert ai_evaluation is not None
                answer.ai_evaluation = ai_evaluation
                answer.is_done_rendering = True
            session.add(answer)
            session.commit()

    except Exception as e:
        print(f"BACKEND:\tBackground evaluation failed: {e}")
        session.rollback()
    finally:
        session.close()
#endregion
# ==============================
