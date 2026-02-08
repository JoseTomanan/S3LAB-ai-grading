from sqlmodel import SQLModel, Field
from typing import Optional


class Section(SQLModel, table=True):
    __tablename__ = "section"
    section_id: int = Field(primary_key=True, index=True)
    section: str = Field(index=True)


class Student(SQLModel, table=True):
    __tablename__ = "student"
    student_no: str = Field(primary_key=True, max_length=50)
    name: str
    section_id: int = Field(foreign_key="section.section_id", index=True)


class TestInstance(SQLModel, table=True):
    __tablename__ = "test_instance"
    test_id: str = Field(primary_key=True, max_length=100)
    name: str
    section_id: int = Field(foreign_key="section.section_id", index=True)
    date: str  # Consider DATE type in migration
    is_done_rendering: bool = False


class TestItem(SQLModel, table=True):
    __tablename__ = "test_item"
    item_id: int = Field(primary_key=True)
    test_id: str = Field(foreign_key="test_instance.test_id", index=True)
    label: str = Field(max_length=50)
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str


class TestPaperInstance(SQLModel, table=True):
    __tablename__ = "test_paper"
    paper_id: int = Field(primary_key=True)
    test_id: str = Field(foreign_key="test_instance.test_id", index=True)
    student_no: str = Field(foreign_key="student.student_no", index=True)
    is_done_rendering: bool = False


class StudentAnswer(SQLModel, table=True):
    __tablename__ = "student_answer"
    answer_id: int = Field(primary_key=True)
    paper_id: int = Field(foreign_key="test_paper.paper_id", index=True)
    item_id: int = Field(foreign_key="test_item.item_id", index=True)
    image_directory: str = Field(default="", max_length=500)
    ai_evaluation: str = Field(default="")
    is_done_rendering: bool = False