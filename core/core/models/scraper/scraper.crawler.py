"""
Módulo de scraping: requisições HTTP com retry automático
e extração de texto via BeautifulSoup.
"""

import time
import logging
import requests

from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_strategy = Retry(
        total=settings.SCRAPER_MAX_RETRIES,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _extract_texts(html: str, css_selector: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.select(css_selector)
    texts = []
    for el in elements:
        text = el.get_text(separator=" ", strip=True)
        if len(text) >= settings.MIN_TEXT_LENGTH:
            texts.append(text)
    return texts


def scrape_urls(urls: list[str], css_selector: str) -> list[dict]:
    """
    Raspa uma lista de URLs e retorna os textos extraídos.
    Retorna lista de dicts: url, status, texts, error_message.
    """
    session = _build_session()
    results = []

    for url in urls:
        logger.info("Raspando: %s", url)
        try:
            response = session.get(url, timeout=settings.SCRAPER_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            texts = _extract_texts(response.text, css_selector)
            results.append({
                "url": url,
                "status": "ok",
                "texts": texts,
                "error_message": None,
            })
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP %s em %s", e.response.status_code, url)
            results.append({
                "url": url,
                "status": "erro",
                "texts": [],
                "error_message": str(e),
            })
        except requests.exceptions.RequestException as e:
            logger.warning("Erro de rede em %s: %s", url, e)
            results.append({
                "url": url,
                "status": "erro",
                "texts": [],
                "error_message": str(e),
            })
        time.sleep(settings.SCRAPER_DELAY)

    return results
