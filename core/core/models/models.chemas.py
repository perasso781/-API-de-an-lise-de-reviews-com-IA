"""
Schemas Pydantic: definem e validam os dados que entram e saem da API.
Qualquer dado fora do formato é rejeitado automaticamente pelo FastAPI.
"""

from pydantic import BaseModel, HttpUrl, field_validator
from typing import Literal


class ScrapeRequest(BaseModel):
    urls: list[HttpUrl]
    css_selector: str = "p"

    @field_validator("urls")
    @classmethod
    def limit_urls(cls, urls: list[HttpUrl]) -> list[HttpUrl]:
        from core.config import get_settings
        max_urls = get_settings().MAX_URLS_PER_REQUEST
        if len(urls) > max_urls:
            raise ValueError(f"Máximo de {max_urls} URLs por requisição.")
        if len(urls) == 0:
            raise ValueError("Informe ao menos uma URL.")
        return urls


class TrainRequest(BaseModel):
    texts: list[str]
    labels: list[Literal["positivo", "negativo", "neutro"]]

    @field_validator("labels")
    @classmethod
    def labels_must_match_texts(cls, labels, values):
        texts = values.data.get("texts", [])
        if len(labels) != len(texts):
            raise ValueError("'texts' e 'labels' devem ter o mesmo tamanho.")
        return labels


class TextAnalysis(BaseModel):
    text: str
    sentiment: Literal["positivo", "negativo", "neutro"]
    confidence: float


class PageResult(BaseModel):
    url: str
    status: Literal["ok", "erro"]
    error_message: str | None = None
    analyses: list[TextAnalysis] = []

    @property
    def overall_sentiment(self) -> str | None:
        if not self.analyses:
            return None
        from collections import Counter
        counts = Counter(a.sentiment for a in self.analyses)
        return counts.most_common(1)[0][0]


class ScrapeResponse(BaseModel):
    results: list[PageResult]
    total_pages: int
    pages_with_error: int


class TrainResponse(BaseModel):
    message: str
    accuracy: float
    samples_trained: int
