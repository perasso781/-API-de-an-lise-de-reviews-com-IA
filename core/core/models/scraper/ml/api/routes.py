"""
Rotas da API.

GET  /api/v1/health   → status (sem auth)
POST /api/v1/analyze  → scrapa URLs e analisa sentimento
POST /api/v1/train    → re-treina o modelo
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from core.security import validate_api_key
from models.schemas import (
    ScrapeRequest, ScrapeResponse, PageResult, TextAnalysis,
    TrainRequest, TrainResponse,
)
from scraper.crawler import scrape_urls
from ml.sentiment import predict, train_model

logger = logging.getLogger(__name__)
router = APIRouter()

_ml_state: dict = {}


def set_ml_state(pipeline, encoder) -> None:
    _ml_state["pipeline"] = pipeline
    _ml_state["encoder"] = encoder


@router.get("/health", tags=["Infra"])
def health_check():
    return {"status": "ok", "model_loaded": "pipeline" in _ml_state}


@router.post(
    "/analyze",
    response_model=ScrapeResponse,
    tags=["Análise"],
    dependencies=[Depends(validate_api_key)],
)
def analyze(request: ScrapeRequest):
    if "pipeline" not in _ml_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo ainda não carregado. Tente novamente em instantes.",
        )

    urls_str = [str(u) for u in request.urls]
    scraped = scrape_urls(urls_str, request.css_selector)

    page_results = []
    errors = 0

    for item in scraped:
        if item["status"] == "erro" or not item["texts"]:
            errors += 1
            page_results.append(PageResult(
                url=item["url"],
                status="erro",
                error_message=item.get("error_message", "Nenhum texto encontrado"),
            ))
            continue

        predictions = predict(
            _ml_state["pipeline"],
            _ml_state["encoder"],
            item["texts"],
        )

        analyses = [
            TextAnalysis(text=text, **pred)
            for text, pred in zip(item["texts"], predictions)
        ]

        page_results.append(PageResult(
            url=item["url"],
            status="ok",
            analyses=analyses,
        ))

    return ScrapeResponse(
        results=page_results,
        total_pages=len(page_results),
        pages_with_error=errors,
    )


@router.post(
    "/train",
    response_model=TrainResponse,
    tags=["ML"],
    dependencies=[Depends(validate_api_key)],
)
def train(request: TrainRequest):
    try:
        pipeline, encoder, accuracy = train_model(request.texts, request.labels)
        set_ml_state(pipeline, encoder)
        return TrainResponse(
            message="Modelo re-treinado com sucesso.",
            accuracy=accuracy,
            samples_trained=len(request.texts),
        )
    except Exception as e:
        logger.error("Erro ao treinar modelo: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha no treinamento: {str(e)}",
        )
