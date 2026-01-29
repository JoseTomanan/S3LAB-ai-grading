from sqlmodel import SQLModel


class TestInstancesBase(SQLModel):
    name: str
    section: str
    date: str
    is_done_rendering: str

class TestInstancesGetResponse(TestInstancesBase):
    test_id: str

class TestInstancesAddRequest(TestInstancesBase):
    test_id: str

class TestInstancesAddResponse(TestInstancesBase):
    ...

