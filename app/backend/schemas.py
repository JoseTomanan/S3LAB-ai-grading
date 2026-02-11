from pydantic import BaseModel, Field
from typing import List, Optional

# ==============================
# Test Item Schemas
# ==============================

class TestItemSummary(BaseModel):
    """Response model for GET /items endpoint (minimal fields per contract)"""
    question: str
    is_problem_solving: bool


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

# ==============================
# Test Instance Schemas
# ==============================

class NewTestInstanceRequest(BaseModel):
    """Request body for POST /test_instances"""
    name: str
    section_id: int
    date: str

class UpdateTestItemSchema(BaseModel):
    """Schema for updating test items within a test instance"""
    item_id: int
    label: str = Field(..., min_length=1)
    question: Optional[str] = Field(None, min_length=1)
    is_problem_solving: Optional[bool] = None
    expected_answer_rubric_questions: str = Field(..., min_length=1)

class UpdateTestInstanceRequest(BaseModel):
    """Request body for PATCH /test_instances/{test_id}"""
    date: Optional[str] = Field(None, min_length=1)
    items: Optional[List[UpdateTestItemSchema]] = None
    

class TestInstanceResponse(BaseModel):
    """
    Full test instance response (used in PATCH and future GET with items)
    """
    name: str
    section_id: int
    date: str
    test_id: str
    is_done_rendering: bool
    items: List[TestItemSummary]

# ==============================
# Student Schemas
# ==============================

class StudentResponse(BaseModel):
    student_no: str
    name: str
    section_id: int

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

class StudentAnswerSummary(BaseModel):
    student_no: str
    name: str
    item_id: int
    label: str
    image_directory: str
    ai_evaluation: str
    is_done_rendering: bool