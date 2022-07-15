
from fastapi import FastAPI
from routes import studentRoutes

app = FastAPI()



app.include_router(studentRoutes.router, prefix='/api/v1')

