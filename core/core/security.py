"""
Módulo de segurança: valida a API Key em cada requisição.
Usa comparação de tempo constante para evitar timing attacks.
"""

import secrets
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from core.config import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key não informada. Use o header X-API-Key.",
        )

    # secrets.compare_digest evita timing attack:
    # o tempo de resposta é sempre o mesmo, independente da chave enviada
    key_is_valid = secrets.compare_digest(
        api_key.encode("utf-8"),
        settings.API_KEY.encode("utf-8"),
    )

    if not key_is_valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida.",
        )

    return api_key
