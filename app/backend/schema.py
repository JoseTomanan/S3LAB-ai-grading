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
    item_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    is_problem_solving: bool
    expected_answer_rubric_questions: str = Field(..., min_length=1)


class NewTestItemResponse(BaseModel):
    """Response for POST /items"""
    items: List[NewTestItemRequest]


class UpdateTestItemRequest(BaseModel):
    """Request body for PATCH endpoint (partial updates)"""
    question: Optional[str] = Field(None, min_length=1)
    is_problem_solving: Optional[bool] = None
    expected_answer_rubric_questions: Optional[str] = Field(None, min_length=1)


class FullTestItemResponse(BaseModel):
    """Full item representation for PATCH response"""
    item_id: str
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


class UpdateTestInstanceRequest(BaseModel):
    date: Optional[str] = Field(None, min_length=1)


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