from sqlmodel import SQLModel, Field
from typing import List, Optional



class TestInstance(SQLModel, table=True):
    test_id: str = Field(primary_key=True)
    section_id: str = Field(foreign_key="section.section")
    name: str = Field()
    date: str = Field()
    is_done_rendering: bool = Field()

class TestPaper(SQLModel, table=True):
    paper_id: int = Field(primary_key=True, index=True)
    test_id: str = Field(foreign_key="testinstance.test_id")
    student_no: str = Field(foreign_key="student.student_no")
    is_done_rendering: bool = Field()

class TestItem(SQLModel, table=True):
    item_id: int = Field(primary_key=True, index=True)
    test_id: str = Field(foreign_key="testinstance.test_id")
    label: str = Field()
    question: str = Field()
    is_problem_solving: bool = Field()
    expected_answer_rubric_questions: str = Field()

class StudentAnswer(SQLModel, table=True):
    answer_id: str = Field(primary_key=True, index=True)
    paper_id: int = Field(foreign_key="testpaper.paper_id")
    item_id: int = Field(foreign_key="testitem.item_id")
    image_directory: str = Field()
    ai_evaluation: str = Field()
    is_done_rendering: bool = Field()

class Student(SQLModel, table=True):
    student_no: str = Field(primary_key=True)
    name: str = Field()
    section: str = Field(foreign_key="section.section")

class Section(SQLModel, table=True):
    section_id: int = Field(primary_key=True, index=True)
    section: str = Field()
