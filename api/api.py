from fastapi import FastAPI

api = FastAPI()

# GET, POST, PUT, and DELETE

@api.get('/')
def index():
    return {"message: Hello World"}