"""
Configurações centrais da aplicação.
Usa pydantic-settings para carregar variáveis de ambiente com segurança.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "Review Intelligence API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- Segurança ---
    API_KEY: str = "troque-essa-chave-antes-de-subir-em-producao"
    API_KEY_HEADER: str = "X-API-Key"

    # --- Scraper ---
    SCRAPER_TIMEOUT: int = 15
    SCRAPER_MAX_RETRIES: int = 3
    SCRAPER_DELAY: float = 1.5

    # --- ML ---
    MODEL_PATH: str = "ml/saved_model"
    MIN_TEXT_LENGTH: int = 10

    # --- Rate limiting simples ---
    MAX_URLS_PER_REQUEST: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
