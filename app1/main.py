from fastapi import FastAPI
from app1.api.routes import router

app = FastAPI()

app.include_router(router)
