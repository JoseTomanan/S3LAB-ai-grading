from sqlmodel import SQLModel, Field
from typing import List, Optional



class TestInstance(SQLModel, table=True):
    test_id: str = Field(primary_key=True)
    name: str = Field()
    section: str = Field()
    date: str = Field()
    is_done_rendering: bool = Field()

class TestPaperInstance(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_no: str = Field(foreign_key="person.student_no")

class TestItem(SQLModel, table=True):
    item_id: str = Field(primary_key=True, index=True)

class Person(SQLModel, table=True):
    student_no: int = Field(primary_key=True, index=True)
    name: str = Field()
    section: str = Field(foreign_key="section.section_name")

class Section(SQLModel, table=True):
    section_name: str = Field(primary_key=True)
