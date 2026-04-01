"""
Ponto de entrada da aplicação.

Para rodar localmente:
  uvicorn main:app --reload --port 8000

Para rodar via Docker:
  docker compose up --build
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from core.config import get_settings
from api.routes import router, set_ml_state
from ml.sentiment import load_or_train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    pipeline, encoder = load_or_train_model()
    set_ml_state(pipeline, encoder)
    logger.info("Modelo pronto. API disponível.")
    yield
    logger.info("Encerrando aplicação.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Bloqueia hosts não autorizados (evita ataques de Host Header)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.seudominio.com"],
)

# CORS: em produção, troque "*" pelo domínio real do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://seudominio.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
