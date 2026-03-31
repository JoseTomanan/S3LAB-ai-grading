from pydantic import BaseModel
from typing import Optional



class StudentResponse(BaseModel):
    student_no: str
    name: str
    section_id: int

class NewStudentRequest(BaseModel):
    student_no: str
    name: str
    section_id: int

class UpdateStudentRequest(BaseModel):
    name: Optional[str] = None
    section_id: Optional[int] = None
