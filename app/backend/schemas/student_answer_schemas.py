from pydantic import BaseModel
from typing import Optional



class StudentAnswerRequest(BaseModel):
    paper_id: int
    item_id: int
    image_directory: str
    ai_evaluation: Optional[str] = ""
    is_done_rendering: bool = False

class StudentAnswerResponse(BaseModel):
    answer_id: int
    paper_id: int
    item_id: int
    image_directory: str
    ai_evaluation: str
    is_done_rendering: bool
    scores: str

class StudentAnswerSummary(BaseModel):
    student_no: str
    name: str
    item_id: int
    label: str
    image_directory: str
    ai_evaluation: str
    is_done_rendering: bool

class BoxesItem(BaseModel):
    index: int
    image_directory: str
    item_number: str

class CommitBoxesRequest(BaseModel):
    boxes: list[BoxesItem]

class ImagePreprocessResponse(BaseModel):
    num_boxes: int
    boxes: list[BoxesItem]

class LabelSaveBoxesResponse(BaseModel):
    num_boxes: int
    boxes: list[BoxesItem]

class UpdateSegmentationResponse(BaseModel):
    image_directory: str

class StudentStatusItem(BaseModel):
    student_no: str
    name: str
    is_done_rendering: bool
    is_all_answers_evaluated: bool
    has_any_answer: bool
    total_score: str

class TestStatusesResponse(BaseModel):
    test_id: str
    statuses: list[StudentStatusItem]

class AIEvaluationItem(BaseModel):
    item_id: int
    answer_id: int
    label: str
    question: str
    expected_answer_rubric_questions: str
    ai_evaluation: str

class AIEvaluationItemWithScores(BaseModel):
    item_id: int
    answer_id: int
    label: str
    question: str
    expected_answer_rubric_questions: str
    ai_evaluation: str
    scores: str

class StudentEvaluationsStore(BaseModel):
    student_no: str
    name: str
    evaluations: list[AIEvaluationItem]

class TestResultsResponse(BaseModel):
    test_id: str
    students: list[StudentEvaluationsStore]

class StudentResultsResponse(BaseModel):
    test_id: str
    student_no: str
    name: str
    evaluations: list[AIEvaluationItemWithScores]
