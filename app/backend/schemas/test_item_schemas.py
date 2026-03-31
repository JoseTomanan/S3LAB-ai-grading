from pydantic import BaseModel, Field
from typing import List, Optional



class TestItemSummary(BaseModel):
    """Response model for GET /items endpoint (minimal fields per contract)"""
    item_id: int
    label: str
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str

class TestItemsResponse(BaseModel):
    test_id: str
    items: List[TestItemSummary]

class NewTestItemRequest(BaseModel):
    """Request body for POST /items (item_id is generated server-side)"""
    label: str = "Item"
    question: str = "Untitled Question"
    is_problem_solving: bool = False
    expected_answer_rubric_questions: str = ""

class NewTestItemResponse(BaseModel):
    """Response for POST /items (returns full created items)"""
    items: List["FullTestItemResponse"]

class UpdateTestItemRequest(BaseModel):
    """Request body for PATCH /items/{item_id} (partial updates)"""
    label: Optional[str] = Field(None, min_length=1)
    question: Optional[str] = Field(None, min_length=1)
    is_problem_solving: Optional[bool] = None
    expected_answer_rubric_questions: Optional[str] = Field(None, min_length=1)

class FullTestItemResponse(BaseModel):
    """Full item representation for PATCH response and POST response"""
    item_id: int
    label: str
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str

class TestPaperResponse(BaseModel):
    paper_id: int
    test_id: str
    student_no: str
    is_done_rendering: bool
