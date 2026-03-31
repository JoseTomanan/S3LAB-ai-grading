from pydantic import BaseModel



class SectionCreate(BaseModel):
    section: str

class SectionResponse(BaseModel):
    section_id: int
    section_name: str
