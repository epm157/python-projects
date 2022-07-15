from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
from config.config import db
from models.StudentModels import StudentModel, UpdateStudentModel

COLLECTION_NAME = 'students'

router = APIRouter(prefix=f'/{COLLECTION_NAME}', tags=['Students'])

@router.get('/', response_description='List all students', response_model=List[StudentModel])
async def list_students():
    students = await db[COLLECTION_NAME].find().to_list(1000)
    return students

@router.get('/{id}', response_description='Get a single student', response_model=StudentModel)
async def get_student(id: str):
    if (student := await db[COLLECTION_NAME].find_one({'_id': id})) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Student with id: {id} not found')
    return student

@router.post('/', response_description='Add new student', response_model=StudentModel)
async def create_student(student: StudentModel=Body(...)):
    student = jsonable_encoder(student)
    new_student = await db[COLLECTION_NAME].insert_one(student)
    created_student = await db[COLLECTION_NAME].find_one({'_id': new_student.inserted_id})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=created_student)

@router.patch('/{id}', response_description='Update a student', response_model=StudentModel)
async def update_student(id: str, student: UpdateStudentModel = Body(...)):
    student = {k: v for k, v in student.dict().items() if v is not None}
    if len(student) < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Please provide fields to update')

    #update_result.modified_count != 1 or
    #update_result =
    #check for update_result.modified_count
    await db[COLLECTION_NAME].update_one({'_id': id}, {'$set': student})
    if (updated_student := await db[COLLECTION_NAME].find_one({'_id': id})) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Student by id: {id} not found')
    return updated_student


@router.delete('/{id}', response_description='Delete a student', response_model=StudentModel)
async def update_student(id: str):
    delete_result = await db[COLLECTION_NAME].delete_one({'_id': id})
    if delete_result.deleted_count == 1:
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Student by id: {id} not found')




