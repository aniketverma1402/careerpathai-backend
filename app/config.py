from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# Project root (careerpathai/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load .env into environment variables
load_dotenv(dotenv_path=ENV_PATH)


class Settings(BaseSettings):
    app_name: str = "CareerPathAI"
    database_path: str = "data/careerpath.db"
    allowed_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
