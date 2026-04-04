import json
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional



class Section(SQLModel, table=True):
    section_id: int = Field(primary_key=True, index=True)
    section: str = Field(index=True)

    # # ======== Relationships ========
    # students: List["Student"] = Relationship(back_populates="section")
    # tests: List["TestInstance"] = Relationship(back_populates="section")



class Student(SQLModel, table=True):
    student_no: str = Field(primary_key=True, max_length=50)
    name: str
    section_id: int = Field(foreign_key="section.section_id", index=True)

    # # ======== Relationships ========
    # section: "Section" = Relationship(back_populates="students")
    # papers: List["TestPaperInstance"] = Relationship(back_populates="student")



class TestInstance(SQLModel, table=True):
    __tablename__ = "test_instance"   # type: ignore[assignment]
    test_id: str = Field(primary_key=True, max_length=100)
    name: str
    section_id: int = Field(foreign_key="section.section_id", index=True)
    date: str 
    is_done_rendering: bool = False

    # # ======== Relationships ========
    # section: "Section" = Relationship(back_populates="tests")
    # items: List["TestItem"] = Relationship(back_populates="test")
    # papers: List["TestPaperInstance"] = Relationship(back_populates="test")



class TestItem(SQLModel, table=True):
    __tablename__ = "test_item"   # type: ignore[assignment]
    item_id: Optional[int] = Field(default=None, primary_key=True)
    test_id: str = Field(foreign_key="test_instance.test_id", index=True)
    label: str = Field(max_length=50)
    question: str
    is_problem_solving: bool
    expected_answer_rubric_questions: str

    # # ======== Relationships ========
    # test: "TestInstance" = Relationship(back_populates="items")
    # answers: List["StudentAnswer"] = Relationship(back_populates="item")



class TestPaperInstance(SQLModel, table=True):
    __tablename__ = "test_paper"   # type: ignore[assignment]
    paper_id: int = Field(default=None, primary_key=True)
    test_id: str = Field(foreign_key="test_instance.test_id", index=True)
    student_no: str = Field(foreign_key="student.student_no", index=True)
    is_done_rendering: bool = False

    # # ======== Relationships ========
    # test: "TestInstance" = Relationship(back_populates="papers")
    # student: "Student" = Relationship(back_populates="papers")
    # answers: List["StudentAnswer"] = Relationship(back_populates="paper")



class StudentAnswer(SQLModel, table=True):
    __tablename__ = "student_answer"   # type: ignore[assignment]
    answer_id: int = Field(default=None, primary_key=True)
    paper_id: int = Field(foreign_key="test_paper.paper_id", index=True)
    item_id: int = Field(foreign_key="test_item.item_id", index=True)
    image_directory: str = Field(default="", max_length=500)
    ai_evaluation: str = Field(default="")
    is_done_rendering: bool = False
    scores: str = Field(default="[]")
    detected_item_number: str = Field(default="UNKNOWN", max_length=10)
    
    # # ======== Relationships ========
    # paper: "TestPaperInstance" = Relationship(back_populates="answers")
    # item: "TestItem" = Relationship(back_populates="answers")

    # ======== Properties ========
    @property
    def scores_list(self) -> List[float]:
        """Get the list of LLM evaluation scores (e.g., [0.9, 1.0, 0.75])."""
        return json.loads(self._scores_json)

    @scores_list.setter
    def scores_list(self, value: List[float]) -> None:
        """Set scores from LLM evaluation; stored as JSON in the database."""
        if not isinstance(value, list):
            raise ValueError("scores must be a list of float numbers")
        # Ensure all elements are float-able
        self._scores_json = json.dumps([float(x) for x in value])