from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.db import init_db
from app.config import get_settings


# Load settings from .env via config.py
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for startup and shutdown events.
    """
    # Startup
    init_db()
    yield
    # Shutdown (nothing needed for now)


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

# Convert comma-separated origins into list
raw_origins: str = settings.allowed_origins
allowed_origins: List[str] = [
    origin.strip() for origin in raw_origins.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": f"{settings.app_name} backend running."}


# API routes
app.include_router(predict_router, prefix="/api")
