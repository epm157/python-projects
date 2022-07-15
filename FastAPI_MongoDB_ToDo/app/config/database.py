from pymongo import MongoClient

client = MongoClient("mongodb+srv://root:123456654321@cluster0.i8zxq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

db = client.todo_app

collection_name = db["todos_app"]