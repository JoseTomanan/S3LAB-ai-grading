# schemas.py
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
    """Request body for POST /items"""
    item_id: int = Field(..., gt=0)  # Changed from str to int
    label: str = Field(..., min_length=1)  # ADDED: label field
    question: str = Field(..., min_length=1)
    is_problem_solving: bool
    expected_answer_rubric_questions: str = Field(..., min_length=1)


class NewTestItemResponse(BaseModel):
    """Response for POST /items"""
    items: List[NewTestItemRequest]


class UpdateTestItemRequest(BaseModel):
    """Request body for PATCH endpoint (partial updates)"""
    label: Optional[str] = Field(None, min_length=1)  # ADDED: label field
    question: Optional[str] = Field(None, min_length=1)
    is_problem_solving: Optional[bool] = None
    expected_answer_rubric_questions: Optional[str] = Field(None, min_length=1)


class FullTestItemResponse(BaseModel):
    """Full item representation for PATCH response"""
    item_id: int  # Changed from str to int
    label: str  # ADDED: label field
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str


# ==============================
# Test Instance Schemas
# ==============================

class NewTestInstanceRequest(BaseModel):
    name: str
    section: str
    date: str

class UpdateTestItemSchema(BaseModel):
    """Schema for updating test items within a test instance"""
    question: str = Field(..., min_length=1)
    is_problem_solving: bool

class UpdateTestInstanceRequest(BaseModel):
    """Request body for PATCH /api/test_instances/{test_id}"""
    date: Optional[str] = Field(None, min_length=1)
    items: Optional[List[UpdateTestItemSchema]] = None

class TestInstanceResponse(BaseModel):
    """
    Full test instance response (used in PATCH and future GET with items)
    """
    name: str
    section: str
    date: str
    test_id: str
    is_done_rendering: bool
    items: List[TestItemSummary]