from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from models import *
from schema import *
from database import *

# ==============================
# Static Test Data (Temporary – Replace with DB Later)
# ==============================

TEST_INSTANCES = [
    {
        "name": "Seatwork-1",
        "section": "3-Rizal",
        "date": "2025-11-11T20:17:46.384Z",
        "test_id": "3-Rizal_Seatwork-1",
        "is_done_rendering": True
    },
    {
        "name": "Seatwork-2",
        "section": "3-Aguinaldo",
        "date": "2025-12-12T20:17:46.384Z",
        "test_id": "3-Aguinaldo_Seatwork-2",
        "is_done_rendering": False
    },
    {
        "name": "Quiz-1",
        "section": "3-Aguinaldo",
        "date": "2026-01-12T20:17:46.384Z",
        "test_id": "3-Aguinaldo_Quiz-1",
        "is_done_rendering": False
    },
    {
        "name": "Quiz-1",
        "section": "3-Rizal",
        "date": "2026-01-13T20:17:46.384Z",
        "test_id": "3-Rizal_Quiz-1",
        "is_done_rendering": True
    },
    {
        "name": "Quiz-2",
        "section": "3-Aguinaldo",
        "date": "2026-01-19T20:17:46.384Z",
        "test_id": "3-Aguinaldo_Quiz-2",
        "is_done_rendering": True
    }
]

# Use a set for O(1) lookup
VALID_TEST_IDS = {inst["test_id"] for inst in TEST_INSTANCES}

# Simulated in-memory storage for test items (to be replaced by DB)
test_items_db = {
    "3-Rizal_Seatwork-1": [
        {
            "item_id": "item_1",
            "question": "Solve for x: 2x + 5 = 15",
            "is_problem_solving": True,
            "expected_answer_rubric_questions": "Correct equation setup (2pts), accurate solution (2pts), proper units (1pt)"
        },
        {
            "item_id": "item_2",
            "question": "What is the capital of France?",
            "is_problem_solving": False,
            "expected_answer_rubric_questions": "Correct answer: Paris (1pt)"
        }
    ],
    "3-Aguinaldo_Quiz-2": [
        {
            "item_id": "item_1",
            "question": "Calculate the area of a circle with radius 7cm",
            "is_problem_solving": True,
            "expected_answer_rubric_questions": "Correct formula (2pts), substitution (1pt), calculation (1pt), units (1pt)"
        }
    ]
}

# ==============================
# Pydantic Models
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
# App Setup
# ==============================

create_db_and_tables()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Endpoints
# ==============================

@app.get("/api/test_instances")
def get_test_instances():
    return {"instances": TEST_INSTANCES}


@app.get(
    "/api/test_instances/{test_id}/items",
    response_model=TestItemsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"description": "Test instance not found"},
        400: {"description": "Invalid test_id format"}
    }
)
def get_test_items(test_id: str):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    items = test_items_db.get(test_id, [])
    summary_items = [
        TestItemSummary(
            question=item["question"],
            is_problem_solving=item["is_problem_solving"]
        )
        for item in items
    ]
    
    return TestItemsResponse(test_id=test_id, items=summary_items)


@app.post(
    "/api/test_instances/{test_id}/items",
    response_model=NewTestItemResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input or duplicate item_id"},
        404: {"description": "Test instance not found"}
    }
)
def add_test_item(test_id: str, item: NewTestItemRequest):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if test_id not in test_items_db:
        test_items_db[test_id] = []
    
    if any(existing["item_id"] == item.item_id for existing in test_items_db[test_id]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Item ID '{item.item_id}' already exists in test instance '{test_id}'"
        )
    
    test_items_db[test_id].append(item.dict())
    return NewTestItemResponse(items=[item])


@app.patch(
    "/api/test_instances/{test_id}/{test_item_id}",
    response_model=FullTestItemResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid ID format"},
        404: {"description": "Test instance or item not found"}
    }
)
def edit_test_item(test_id: str, test_item_id: str, update: UpdateTestItemRequest):
    if not test_id or len(test_id) > 100 or not test_item_id or len(test_item_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if test_id not in test_items_db or not any(
        item["item_id"] == test_item_id for item in test_items_db[test_id]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test item with ID '{test_item_id}' not found in test instance '{test_id}'"
        )
    
    for item in test_items_db[test_id]:
        if item["item_id"] == test_item_id:
            update_data = update.dict(exclude_unset=True)
            item.update(update_data)
            return FullTestItemResponse(**item)
    
    # Fallback (should not occur)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Item update failed unexpectedly"
    )


@app.delete(
    "/api/test_instances/{test_id}/{test_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid ID format"},
        404: {"description": "Test instance or item not found"}
    }
)
def delete_test_item(test_id: str, test_item_id: str):
    if not test_id or len(test_id) > 100 or not test_item_id or len(test_item_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    if test_id not in test_items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No items found for test instance '{test_id}'"
        )
    
    original_count = len(test_items_db[test_id])
    test_items_db[test_id] = [
        item for item in test_items_db[test_id] if item["item_id"] != test_item_id
    ]
    
    if len(test_items_db[test_id]) == original_count:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test item with ID '{test_item_id}' not found in test instance '{test_id}'"
        )
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)