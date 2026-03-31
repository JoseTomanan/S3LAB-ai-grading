from fastapi import APIRouter, HTTPException, Depends, status, Response
from sqlmodel import Session, delete, select

from models import *
from schemas import *
from core.database import get_session



router = APIRouter()


#region Endpoints
@router.post("/", response_model=StudentResponse, status_code=status.HTTP_201_CREATED)
def add_new_student(
                student_data: NewStudentRequest,
                session: Session = Depends(get_session)
                ):
    """Add new student (section_id provided in body per contract)"""
    # Verify section exists FIRST (critical FK validation)
    section = session.get(Section, student_data.section_id)
    if not section:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {student_data.section_id} not found"
                )

    # Check for duplicate student ID
    existing = session.exec(
            select(Student).where(Student.student_no == student_data.student_no)
            ).first()
    if existing:
        raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Student with ID '{student_data.student_no}' already exists"
                )

    new_student = Student(
            student_no=student_data.student_no,
            name=student_data.name,
            section_id=student_data.section_id,
            )
    session.add(new_student)
    session.commit()
    session.refresh(new_student)

    return StudentResponse(
            student_no=new_student.student_no,
            name=new_student.name,
            section_id=new_student.section_id,
            )


@router.patch("/{student_no}", response_model=StudentResponse)
def edit_student_details(
                student_no: str,
                update_data: UpdateStudentRequest,
                session: Session = Depends(get_session)
                ):
    """Edit student details"""
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
                )

    if update_data.name is not None:
        student.name = update_data.name

    if update_data.section_id is not None:
        section = session.get(Section, update_data.section_id)
        if not section:
            raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Section with ID {update_data.section_id} not found"
                    )
        student.section_id = update_data.section_id

    session.add(student)
    session.commit()
    session.refresh(student)

    return StudentResponse(
            student_no=student.student_no,
            name=student.name,
            section_id=student.section_id
            )


@router.delete("/{student_no}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(student_no: str, session: Session = Depends(get_session)):
    """Delete student"""
    student = session.get(Student, student_no)
    if not student:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with ID '{student_no}' not found"
                )
    
    session.delete(student)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

#endregion