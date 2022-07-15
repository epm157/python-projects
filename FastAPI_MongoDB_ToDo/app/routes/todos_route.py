from fastapi import APIRouter

from bson import ObjectId

from app.config.database import collection_name
from app.models.todos_model import ToDo
from app.schemas.todos_schema import todos_serializer, todo_serializer

todo_api_router = APIRouter()

@todo_api_router.get('/')
async def get_todos():
    todos = todos_serializer(collection_name.find())
    return todos

@todo_api_router.get('/{id}')
async def get_todo(id: str):
    return todo_serializer(collection_name.find_one({'_id': ObjectId(id)}))

@todo_api_router.post('/')
async def create_todo(todo: ToDo):
    _id = collection_name.insert_one(dict(todo))
    return todo_serializer(collection_name.find_one({'_id': _id.inserted_id}))

@todo_api_router.patch('/')
async def update_todo(id: str, todo: ToDo):
    collection_name.find_one_and_update({'_id': ObjectId(id)}, {'$set': dict(todo)})
    return todo_serializer(collection_name.find_one({'_id': ObjectId(id)}))

@todo_api_router.delete('/{id}')
async def delete_todo(id: str):
    collection_name.find_one_and_delete({'_id': ObjectId(id)})
    return {'status': 'OK'}






