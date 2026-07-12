"""Application configuration."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base configuration shared by the application."""

    SECRET_KEY = "change-me-in-production"
    DATASET_DIR = BASE_DIR / "dataset"
    MODEL_DIR = BASE_DIR / "models"
    UPLOAD_DIR = BASE_DIR / "static" / "uploads"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'database' / 'retinaai.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
