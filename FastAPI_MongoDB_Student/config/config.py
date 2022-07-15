import os
import motor.motor_asyncio

MONGODB_URL = 'mongodb+srv://root:123456654321@cluster0.i8zxq.mongodb.net/Student?retryWrites=true&w=majority'
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client.college