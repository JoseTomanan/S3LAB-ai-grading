from fastapi import HTTPException, status
from sqlmodel import Session, delete, select

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
#region Private functions
def _mapp(h):
    """
    Reorders and reorganizes a set of 4 points (corners) based on their spatial properties.
    
    Takes a flattened array of 4 points (8 coordinates) and reshapes it to (4, 2),
    then reorders them by:
    - Index 0: Point with minimum sum of coordinates (top-left area)
    - Index 1: Point with minimum difference between x and y (leftmost)
    - Index 2: Point with maximum sum of coordinates (bottom-right area)
    - Index 3: Point with maximum difference between x and y (rightmost)
    
    Args:
        h: Array-like of shape (8,) containing 4 points as flattened coordinates
        
    Returns:
        np.ndarray: Reordered points of shape (4, 2) with dtype float32
    """
    h = h.reshape((4, 2))
    hnew = np.zeros(
                (4, 2),
                dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def _get_robust_aspect_ratio(coords):
    """
    Calculate the aspect ratio of a quadrilateral defined by its corner coordinates.
    This function takes a set of coordinates representing the corners of a quadrilateral,
    sorts them by angle relative to their center point, and computes the aspect ratio
    (width to height) by averaging the lengths of opposite sides.
    Args:
        coords: Array-like of shape (4, 2) containing the (x, y) coordinates of the
                quadrilateral's four corners.
    Returns:
        float: The aspect ratio (width / height) of the quadrilateral.
    """
    pts = np.array(coords)

    center = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    
    side_a1 = pts[1] - pts[0]
    side_a2 = pts[3] - pts[2]
    side_b1 = pts[2] - pts[1]
    side_b2 = pts[0] - pts[3]

    w = (np.linalg.norm(side_a1) + np.linalg.norm(side_a2)) / 2
    h = (np.linalg.norm(side_b1) + np.linalg.norm(side_b2)) / 2

    return w / h


def _is_valid_quad(pts: np.ndarray) -> bool:
    """
    Check if a set of four points forms a valid section (quadrilateral)
    based on area, aspect ratio, and skew angle.

    This function ALLOWS non-perfect quadrilaterals. It considers a quadrilateral
    valid if it approximately meets the size, shape, and skew constraints—i.e.,
    the four points do not need to form a mathematically perfect rectangle or square,
    but must "closely" resemble one within tolerance defined by constants.

    Args:
        pts (np.ndarray): 4x2 array of corner points.

    Returns:
        bool: True if the points approximate a valid quadrilateral section, False otherwise.
    """
    ordered = _mapp(pts.flatten())
    tl, tr, br, bl = ordered
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    if w * h < MIN_AREA:
        return False

    aspect = _get_robust_aspect_ratio(pts)
    if aspect > MAX_ASPECT_RATIO or aspect < 1/MAX_ASPECT_RATIO:
        return False

    angle_top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    angle_bot = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    if abs(angle_top - angle_bot) > MAX_SKEW_DEG:
        return False
    return True
#endregion
# ==============================



# ==============================
#region Auxiliary Functions
def evaluate_image_logic(answer_id_input: int, session: Session):
    _STRIP_POINTS = lambda x : re.sub(r'\s*\([^)]*\)\s*$', '', x).strip()
    _VALID_R_Q_RESPONSE = lambda x : x in ["YES", "NO"]
    _VALID_E_A_RESPONSE = lambda x : x in ["YES", "NO", "UNCLEAR"]

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

    ai_evaluation = ""
    match test_item.is_problem_solving:
        case True:
            rubric_questions = test_item.expected_answer_rubric_questions.split(";")
            for rubric in rubric_questions:
                if rubric.strip() != "":
                    while True:
                        response = AI_ANSWER_EVALUATOR.evaluate_rubric(image_bytes, test_item.question, _STRIP_POINTS(rubric))
                        if response and _VALID_R_Q_RESPONSE(response):
                            break
                    ai_evaluation += f"{response};"
    
        case _:
            expected_answer = test_item.expected_answer_rubric_questions
            while True:
                response = AI_ANSWER_EVALUATOR.evaluate_expected_answer(image_bytes, test_item.question, _STRIP_POINTS(expected_answer))
                if response and _VALID_E_A_RESPONSE(response):
                    break
            
            ai_evaluation = response

    all_scores = calculate_score(test_item.expected_answer_rubric_questions,
                                 ai_evaluation)
    
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
    
    enhanced = IMAGE_MODIFIER.brighten(warped, amount=0.2)
    enhanced = IMAGE_MODIFIER.adjust_contrast(enhanced, amount=1.2)
    
    success, buffer = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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

    SHEETS_EXPORTER = SheetsExporter(columns=test_items_labels)

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
        rowKey = s.name
        print(f"INTERNAL:\tEvaluating {rowKey}...")
        SHEETS_EXPORTER.add_student(rowKey)
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