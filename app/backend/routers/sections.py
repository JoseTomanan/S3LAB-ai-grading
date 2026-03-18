from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel import Session, delete, select

from models import *
from schemas import *

from core.database import get_session


router = APIRouter()


#region Endpoints
@router.get("/", response_model=List[dict])
def get_all_sections(session: Session = Depends(get_session)):
    """Get all sections"""
    sections = session.exec(select(Section)).all()
    return [{"section_id": s.section_id, "section_name": s.section} for s in sections]


@router.get("/{section_id}", response_model=List[StudentResponse])
def get_students_in_section(section_id: int, session: Session = Depends(get_session)):
    """Get all students from a specific section"""
    section = session.get(Section, section_id)
    if not section:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {section_id} not found"
                )
    
    # Get students in this section
    students = session.exec(
            select(Student).where(Student.section_id == section_id)
            ).all()
    
    return [
        StudentResponse(
            student_no=s.student_no,
            name=s.name,
            section_id=s.section_id
        ) for s in students
        ]


@router.post("/", status_code=status.HTTP_201_CREATED)
def add_new_section(section: SectionCreate, session: Session = Depends(get_session)):
    """Add a new section"""
    # Check if section with this name already exists
    existing = session.exec(select(Section).where(Section.section == section.section)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Section '{section.section}' already exists."
        )

    new_section = Section(section=section.section)
    session.add(new_section)
    session.commit()
    session.refresh(new_section)
    return {"section_id": new_section.section_id, "section_name": new_section.section}
#endregion