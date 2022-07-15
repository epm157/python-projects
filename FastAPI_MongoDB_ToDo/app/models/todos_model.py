from pydantic import BaseModel

class ToDo(BaseModel):
    name: str
    description: str
    completed: bool
    date: str

