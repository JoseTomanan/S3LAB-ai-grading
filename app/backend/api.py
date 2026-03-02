from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from models import *
from schemas import *
from database import create_db_and_tables, get_session

from services.utility import *

from routers import sections, students, utility, test_instances, test_items, student_answers



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
app.include_router(utility.router, prefix="/api", tags=["Utility"])

