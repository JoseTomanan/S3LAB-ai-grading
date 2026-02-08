from sqlmodel import SQLModel, Field
from typing import Optional


class Section(SQLModel, table=True):
    section_name: str = Field(primary_key=True)


class Person(SQLModel, table=True):
    student_no: int = Field(primary_key=True, index=True)
    name: str
    section: str = Field(foreign_key="section.section_name")


class TestInstance(SQLModel, table=True):
    test_id: str = Field(primary_key=True)
    name: str
    section: str
    date: str
    is_done_rendering: bool = False


class TestItem(SQLModel, table=True):
    item_id: int = Field(primary_key=True, index=True)
    test_id: str = Field(foreign_key="testinstance.test_id")
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str
    label: str


class TestPaperInstance(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_no: str = Field(foreign_key="person.student_no")