from pydantic import BaseModel, Field
from typing import List, Optional

from .test_item_schemas import TestItemSummary



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
    """Full test instance response (used in PATCH and future GET with items)"""
    name: str
    section_id: int
    date: str
    test_id: str
    is_done_rendering: bool
    items: List[TestItemSummary]
