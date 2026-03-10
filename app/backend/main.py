from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pathlib import Path

from models import *
from schemas import *
from database import create_db_and_tables, get_session

from logic.utility import *

from routers import sections, students, test_instances, test_items, student_answers



#region App Initialization
app = FastAPI(
        title="Assessment Processing API",
        description="API for managing test instances, items, and student answer processing",
        lifespan=create_db_and_tables(),
        version="1.0.0"
        )

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )

TEMP_DIR = Path("static/images")
TEMP_DIR.mkdir(exist_ok=True)
#endregion


app.include_router(test_instances.router, prefix="/api/test_instances", tags=["Test Instances"])
app.include_router(test_items.router, prefix="/api/test_items", tags=["Test Items"])
app.include_router(students.router, prefix="/api/students", tags=["Students"])
app.include_router(student_answers.router, prefix="/api/student_answers", tags=["Student Answers"])
app.include_router(sections.router, prefix="/api/sections", tags=["Sections"])


#region Endpoints
@app.get("/api/temp/{filename}")
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


@app.patch("/api/answers/{answer_id}/reevaluate")
async def reevaluate_answer(answer_id: int, session: Session = Depends(get_session)):
    """Re-evaluate image then store to StudentAnswer evaluation result."""
    # TODO: 400 handling
    # TODO: 404 handling
    return evaluate_image_logic(
                answer_id_input=answer_id,
                session=session
                )
#endregion
