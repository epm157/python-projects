from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional

from models.PyObjectId import PyObjectId

class StudentModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias='_id')
    name: str = Field(...)
    email: EmailStr = Field(...)
    course: str = Field(...)
    gpa: float = Field(..., le=5.0)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            'example': {
                'name': 'Jane Doe',
                'email': 'janedoe@example.com',
                'course': 'Experiments, Science, and Fashion in Nanophotonics',
                'gpa': '3.0'
            }
        }

class UpdateStudentModel(BaseModel):
    name: Optional[str]
    email: Optional[EmailStr]
    course: Optional[str]
    gpa: Optional[float]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            'example': {
                'name': 'Jane Doe',
                'email': 'janedoe@example.com',
                'course': 'Experiments, Science, and Fashion in Nanophotonics',
                'gpa': '3.0'
            }
        }