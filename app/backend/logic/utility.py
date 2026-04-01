from fastapi import HTTPException, status
from sqlmodel import Session, select

import numpy as np
import cv2
import re
import openpyxl

from models import *
from schemas import *
from core.constants import *
from logic.ai_interface import AIAnswerEvaluator
from logic.sheets_exporter import SheetsExporter
from logic.box_segmenter import BoxSegmenter
from logic.document_scanner import DocumentScanner
from logic.image_modifier import ImageModifier


AI_ANSWER_EVALUATOR = AIAnswerEvaluator()
DOCUMENT_SCANNER = DocumentScanner()
IMAGE_MODIFIER = ImageModifier()



# ==============================
#region Auxiliary Functions
def compute_ai_evaluation(image_bytes: bytes, is_problem_solving: bool, question: str, expected_answer_rubric_questions: str) -> str:
    """Run AI evaluation for a single answer. Pure AI call — no DB access.
    Accepts primitives instead of a TestItem ORM object so it is safe to call from worker threads."""
    _STRIP_POINTS = lambda x : re.sub(r'\s*\([^)]*\)\s*$', '', x).strip()
    _VALID_R_Q_RESPONSE = lambda x : x in ["YES", "NO"]
    _VALID_E_A_RESPONSE = lambda x : x in ["YES", "NO", "UNCLEAR"]

    ai_evaluation = ""
    match is_problem_solving:
        case True:
            rubric_questions = expected_answer_rubric_questions.split(";")
            cleaned_rubrics = [_STRIP_POINTS(r) for r in rubric_questions if r.strip() != ""]

            BATCH_SIZE = 4
            batches = [cleaned_rubrics[i:i+BATCH_SIZE] for i in range(0, len(cleaned_rubrics), BATCH_SIZE)]

            all_parts = []
            for batch in batches:
                for attempt in range(MAX_RUBRIC_RETRIES):
                    response = AI_ANSWER_EVALUATOR.evaluate_multi_rubric(
                                    image_bytes, question, batch
                                    )
                    if response:
                        parts = response.strip().split(";")
                        if len(parts) == len(batch) and all(_VALID_R_Q_RESPONSE(p) for p in parts):
                            all_parts.extend(parts)
                            break
                    print(f"BACKEND:\tRetrying rubric eval (attempt {attempt + 1}/{MAX_RUBRIC_RETRIES})...")
                else:
                    ## All retries exhausted — fill batch with NO so grading can continue
                    print(f"BACKEND:\tRubric eval failed after {MAX_RUBRIC_RETRIES} attempts; defaulting to NO for {len(batch)} rubric(s)")
                    all_parts.extend(["NO"] * len(batch))

            ai_evaluation = ";".join(all_parts) + ";"

        case _:
            expected_answer = expected_answer_rubric_questions
            response = None
            for attempt in range(MAX_ANSWER_RETRIES):
                response = AI_ANSWER_EVALUATOR.evaluate_expected_answer(image_bytes, question, _STRIP_POINTS(expected_answer))
                if response and _VALID_E_A_RESPONSE(response):
                    break
                print(f"BACKEND:\tRetrying answer eval (attempt {attempt + 1}/{MAX_ANSWER_RETRIES})...")
            else:
                ## All retries exhausted — default to UNCLEAR so grading can continue
                print(f"BACKEND:\tAnswer eval failed after {MAX_ANSWER_RETRIES} attempts; defaulting to UNCLEAR")
                response = "UNCLEAR"

            ai_evaluation = response

    return ai_evaluation


def evaluate_image_logic(answer_id_input: int, session: Session):
    print(f"INTERNAL:\tFunction evaluate_image({answer_id_input}) is being executed...")

    answer = session.exec(
                    select(StudentAnswer)
                    .where(StudentAnswer.answer_id == answer_id_input)
                    ).first()
    assert answer is not None

    actual_image_path = f"static/images/{answer.image_directory.split("/")[3]}"
    image_bytes: bytes = DOCUMENT_SCANNER.load_image(actual_image_path)

    test_item = session.exec(
                    select(TestItem)
                    .where(TestItem.item_id == answer.item_id)
                    ).first()
    assert test_item is not None

    ai_evaluation = compute_ai_evaluation(image_bytes, test_item.is_problem_solving, test_item.question, test_item.expected_answer_rubric_questions)
    all_scores = calculate_score(test_item.expected_answer_rubric_questions, ai_evaluation)

    answer.ai_evaluation = ai_evaluation
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
            is_done_rendering=answer.is_done_rendering,
            scores=all_scores,
            )


def calculate_score(expected_answer_rubric_questions: str, ai_evaluation: str) -> str:
    scores = ""
    if ai_evaluation:
        splitted_e_a_r_q = expected_answer_rubric_questions.split(";")
        splitted_evals = ai_evaluation.split(";")
        print(f"INTERNAL:\tCalculating w/ splitted EARQ: {splitted_e_a_r_q}.")

        for i, e_a_r_q in enumerate(splitted_e_a_r_q):
            if e_a_r_q.strip() != "":
                index_start = e_a_r_q.find("[")
                index_end = index_start + e_a_r_q[index_start:].find("p")

                print(f"INTERNAL:\t ---> EARQ: {e_a_r_q}")
                print(f"INTERNAL:\t ---> BECOMES {e_a_r_q[index_start+1:index_end]}")
                
                points = e_a_r_q[index_start+1 : index_end]
                if splitted_evals[i] != "":
                    scores += f"{points}/{points};" if splitted_evals[i] == "YES" \
                                    else f"0/{points};"

        print(f"INTERNAL:\tScore is {scores}")

    return scores


def get_max_score(expected_answer_rubric_questions: str) -> int | float:
    max_score = 0
    for part in expected_answer_rubric_questions.split(";"):
        part = part.strip()
        if part:
            idx_start = part.find("[")
            idx_end = idx_start + part[idx_start:].find("p")
            if idx_start != -1 and idx_end > idx_start:
                points = part[idx_start + 1 : idx_end]
                val = float(points)
                max_score += int(val) if val % 1 == 0 else val
    return max_score


def get_total_score(expected_answer_rubric_questions: str, ai_evaluation: str) -> tuple[int|float, int|float]:
    total_score = 0
    max_score = 0

    scores = calculate_score(expected_answer_rubric_questions, ai_evaluation)
    splitted_scores = scores.split(";")
    print(f"INTERNAL:\tOBTAINED SCORES: {splitted_scores}")

    for s in splitted_scores:
        if s != "":
            grade, total = s.split("/")
            total_score += int(grade) if float(grade)%1==0 else float(grade)
            max_score += int(total) if float(total)%1==0 else float(total)

    return total_score, max_score


def crop_image(contents: bytes, points_data: dict):
    GET_ASPECT_RATIO = lambda pts : \
                        (np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[3] - pts[2])) \
                            / (np.linalg.norm(pts[0] - pts[3]) + np.linalg.norm(pts[1] - pts[2]) + 1e-5)

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
    
    aspect_ratio = GET_ASPECT_RATIO(src_pts)
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
    
    success, buffer = cv2.imencode('.jpg', warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError("Failed to encode processed image")
    
    img_bytes = buffer.tobytes()

    return img_bytes


def populate_spreadsheet_logic(test_id_input: str, session: Session) -> openpyxl.Workbook:
    test_items = session.exec(
                            select(TestItem)
                            .where(TestItem.test_id == test_id_input)
                            ).all()
    test_items_labels = [item.label for item in test_items]

    max_scores = {item.label: get_max_score(item.expected_answer_rubric_questions) for item in test_items}
    SHEETS_EXPORTER = SheetsExporter(columns=test_items_labels, max_scores=max_scores)

    test_instance = session.exec(
                            select(TestInstance)
                            .where(TestInstance.test_id == test_id_input)
                            ).first()
    assert test_instance is not None # FIXME: Make this a 400 handling
    
    students = session.exec(
                        select(Student)
                        .where(Student.section_id == test_instance.section_id)
                        ).all()
    assert students is not None
    
    for s in students:
        rowKey = s.student_no
        print(f"INTERNAL:\tEvaluating {rowKey}...")
        SHEETS_EXPORTER.add_student(rowKey, s.name)
        paper = session.exec(
                            select(TestPaperInstance)
                            .where(TestPaperInstance.student_no == s.student_no)
                            ).first()
        
        if paper is None:
            answers = []
        else:
            answers = session.exec(
                                select(StudentAnswer)
                                .where(StudentAnswer.paper_id == paper.paper_id)
                                ).all()
        
        scoresList: dict[str, float|None] = {}
        for testItem in test_items:
            if testItem.item_id in [a.item_id for a in answers]:
                _tempList = [a for a in answers if a.item_id == testItem.item_id]
                respectiveAnswer = _tempList[0] if _tempList else None

                if respectiveAnswer and respectiveAnswer.ai_evaluation:
                    score, _ = get_total_score(testItem.expected_answer_rubric_questions,
                                                respectiveAnswer.ai_evaluation)
                    scoresList[testItem.label] = score if score else 0
                else:
                    scoresList[testItem.label] = None
            else:
                scoresList[testItem.label] = None

        SHEETS_EXPORTER.append(rowKey, scoresList)

    return SHEETS_EXPORTER.export_sheet()

#endregion
# ==============================