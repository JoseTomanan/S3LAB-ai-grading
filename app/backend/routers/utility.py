from fastapi import APIRouter, HTTPException, Depends, status, Body, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, delete, select
from pathlib import Path

from models import *
from schemas import *
from services import *

from database import get_session



router = APIRouter()
TEMP_DIR = Path("static/images")



#region Endpoints
@router.get("/temp/{filename}")
async def get_processed_image(filename: str):
    """Serve processed images from temp directory"""
    # Security validation
    if not filename.endswith(".jpg") or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Allow only alphanumeric + safe characters
    if not all(c.isalnum() or c in "._-" for c in filename):
        raise HTTPException(status_code=400, detail="Invalid filename characters")
    
    filepath = TEMP_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return FileResponse(filepath, media_type="image/jpeg")


@router.patch("/answers/{answer_id}/reevaluate")
async def reevaluate_answer(answer_id: int, session: Session = Depends(get_session)):
    """Re-evaluate image then store to StudentAnswer evaluation result."""
    # TODO: 400 handling
    # TODO: 404 handling
    return evaluate_image_logic(
                answer_id_input=answer_id,
                session=session
                )
#endregion