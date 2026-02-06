# models.py
from sqlmodel import SQLModel, Field
from typing import Optional


class Section(SQLModel, table=True):
    __tablename__ = "section"
    
    section_name: str = Field(primary_key=True)


class Person(SQLModel, table=True):
    __tablename__ = "person"
    
    student_no: int = Field(primary_key=True, index=True)
    name: str
    section: str = Field(foreign_key="section.section_name")


class TestInstance(SQLModel, table=True):
    __tablename__ = "test_instance"
    
    test_id: str = Field(primary_key=True)
    name: str
    section: str
    date: str
    is_done_rendering: bool = False


class TestItem(SQLModel, table=True):
    __tablename__ = "test_item"

    item_id: str = Field(primary_key=True, index=True)
    test_id: str = Field(foreign_key="test_instance.test_id")  # Now matches __tablename__
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str
    label: Optional[str] = None


class TestPaperInstance(SQLModel, table=True):
    __tablename__ = "test_paper_instance"
    
    id: str = Field(primary_key=True)
    student_no: str = Field(foreign_key="person.student_no")