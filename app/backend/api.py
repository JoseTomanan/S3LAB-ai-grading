from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from models import TestInstance, TestItem, Person, Section, TestPaperInstance
from schemas import (
    TestItemSummary,
    TestItemsResponse,
    NewTestItemRequest,
    NewTestItemResponse,
    UpdateTestItemRequest,
    FullTestItemResponse,
    NewTestInstanceRequest,
    UpdateTestInstanceRequest,
    TestInstanceResponse,
)
from database import *

from fastapi import File, UploadFile
from image_preprocessor import CVImagePreprocessor, CVProcessingError
import mimetypes

import uuid
import os
from pathlib import Path
from fastapi.responses import FileResponse

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

# For Test Instances
@app.get("/api/test_instances")
def get_test_instances():
    return {"instances": TEST_INSTANCES}

@app.get(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def get_test_instance(test_id: str):
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    instance = next((inst for inst in TEST_INSTANCES if inst["test_id"] == test_id), None)
    if instance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    return instance


@app.post(
    "/api/test_instances",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Missing required fields"},
        409: {"description": "Test instance already exists"}
    }
)
def add_test_instance(request: NewTestInstanceRequest):  # <-- typed!
    name = request.name
    section = request.section
    date = request.date
    
    if not name or not section or not date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: name, section, and date"
        )
    
    test_id = f"{section}_{name}"
    if test_id in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Test instance with ID '{test_id}' already exists"
        )
    
    new_instance = {
        "name": name,
        "section": section,
        "date": date,
        "test_id": test_id,
        "is_done_rendering": False
    }
    
    TEST_INSTANCES.append(new_instance)
    VALID_TEST_IDS.add(test_id)
    # test_items_db will be initialized on first item POST
    
    return new_instance


@app.patch(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid test_id format or update payload"},
        404: {"description": "Test instance not found"}
    }
)
def edit_test_instance(test_id: str, update: UpdateTestInstanceRequest):
    # Validate test_id format
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    # Check existence
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    # Ensure only 'date' is being updated
    if update.date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'date' can be updated for a test instance"
        )
    
    # Update the instance in TEST_INSTANCES
    target_instance = None
    for inst in TEST_INSTANCES:
        if inst["test_id"] == test_id:
            inst["date"] = update.date
            target_instance = inst
            break
    
    # Build items summary (even if empty)
    items = test_items_db.get(test_id, [])
    summary_items = [
        TestItemSummary(
            question=item["question"],
            is_problem_solving=item["is_problem_solving"]
        )
        for item in items
    ]

    assert target_instance is not None

    return TestInstanceResponse(
        name=target_instance["name"],
        section=target_instance["section"],
        date=target_instance["date"],
        test_id=target_instance["test_id"],
        is_done_rendering=target_instance["is_done_rendering"],
        items=summary_items
    )

@app.delete(
    "/api/test_instances/{test_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"description": "Invalid test_id format"},
        404: {"description": "Test instance not found"}
    }
)
def delete_test_instance(test_id: str):
    # Validate test_id format
    if not test_id or len(test_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test_id format"
        )
    
    # Check existence
    if test_id not in VALID_TEST_IDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance with ID '{test_id}' not found"
        )
    
    # Remove from TEST_INSTANCES
    global TEST_INSTANCES
    TEST_INSTANCES = [inst for inst in TEST_INSTANCES if inst["test_id"] != test_id]
    VALID_TEST_IDS.discard(test_id)
    
    # Cascade delete: remove all associated test items
    test_items_db.pop(test_id, None)  # Safe deletion (no KeyError if missing)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# For Test Items

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


# ==============================
# Computer Vision Preprocessing Endpoint (Multi-Box)
# ==============================



# Temporary directory for processed images
TEMP_DIR = Path("temp_cv_output")
TEMP_DIR.mkdir(exist_ok=True)

@app.post(
    "/api/test_instances/image_preprocess",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid input (no file, empty file, or invalid format)"},
        415: {"description": "Unsupported media type (not JPEG/PNG)"},
        500: {"description": "CV processing failed"}
    }
)
async def image_preprocess(file: UploadFile = File(...)):
    """
    Process raw student assessment image through CV pipeline.
    Detects up to 3 document-like regions, applies brightening + contrast.
    Returns metadata with URLs to access each processed box.
    """
    # --- Validation ---
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    if not CVImagePreprocessor.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format. Allowed: .jpg, .jpeg, .png. Got: {file.filename}"
        )
    
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}"
        )

    # --- Processing ---
    try:
        preprocessor = CVImagePreprocessor()
        processed_list = preprocessor.process_assessment_image(contents)  # List[bytes], 1–3 items
    except CVProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV processing failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during image processing: {str(e)}"
        )

    # --- Save to temp files and generate URLs ---
    session_id = str(uuid.uuid4())
    boxes_info = []

    for i, img_bytes in enumerate(processed_list):
        filename = f"{session_id}_{i}.jpg"
        filepath = TEMP_DIR / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        boxes_info.append({
            "index": i,
            "image_path": f"/api/temp/{filename}"
        })

    return {
        "num_boxes": len(processed_list),
        "boxes": boxes_info
    }

# --- Serve temporary processed images ---
@app.get("/api/temp/{filename}")
async def get_processed_image(filename: str):
    if not filename.endswith(".jpg") or ".." in filename or not filename.replace(".jpg", "").replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = TEMP_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Processed image not found or expired")
    
    return FileResponse(filepath, media_type="image/jpeg")