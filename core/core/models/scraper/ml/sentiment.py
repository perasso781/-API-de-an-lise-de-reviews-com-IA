"""
Análise de sentimento com scikit-learn.
Pipeline: TF-IDF Vectorizer → Logistic Regression.
"""

import os
import logging
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MODEL_FILE = os.path.join(settings.MODEL_PATH, "sentiment_pipeline.joblib")
ENCODER_FILE = os.path.join(settings.MODEL_PATH, "label_encoder.joblib")
LABELS = ["positivo", "negativo", "neutro"]

_SEED_TEXTS = [
    "Produto excelente, superou todas as expectativas, recomendo muito",
    "Entrega rápida e o item chegou em perfeito estado, adorei",
    "Ótima qualidade, vale cada centavo, comprarei novamente",
    "Serviço de atendimento incrível, resolveram tudo rapidamente",
    "Estou muito satisfeito com a compra, produto de alta qualidade",
    "Funciona perfeitamente, exatamente como descrito na página",
    "Maravilhoso, melhor compra que fiz este ano, parabéns",
    "Produto horrível, quebrou na primeira semana de uso, decepção",
    "Entrega demorou demais e o item veio danificado, péssimo",
    "Não funciona como prometido, propaganda enganosa, vou reclamar",
    "Atendimento terrível, ignoraram minha reclamação completamente",
    "Produto de péssima qualidade, não vale o preço cobrado, lixo",
    "Veio com defeito de fábrica, devolvi e não recebi o reembolso",
    "Experiência horrível, nunca mais compro nesta loja, cuidado",
    "Produto dentro do esperado, nada de especial mas funciona",
    "Entrega no prazo, produto razoável, poderia ser melhor",
    "É o que aparece na foto, nem mais nem menos, ok",
    "Preço justo para o que é, produto mediano sem diferenciais",
    "Aceitável, faz o que promete de forma básica, sem surpresas",
    "Não é ruim, não é bom, apenas cumpre o objetivo, serve",
]
_SEED_LABELS = ["positivo"] * 7 + ["negativo"] * 7 + ["neutro"] * 6


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5_000,
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def _ensure_model_dir() -> None:
    os.makedirs(settings.MODEL_PATH, exist_ok=True)


def load_or_train_model() -> tuple[Pipeline, LabelEncoder]:
    _ensure_model_dir()
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        logger.info("Carregando modelo de %s", MODEL_FILE)
        return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE)
    logger.info("Treinando com dados de seed...")
    pipeline, encoder, _ = train_model(_SEED_TEXTS, _SEED_LABELS)
    return pipeline, encoder


def train_model(
    texts: list[str],
    labels: list[str],
) -> tuple[Pipeline, LabelEncoder, float]:
    _ensure_model_dir()
    encoder = LabelEncoder()
    encoder.fit(LABELS)
    y = encoder.transform(labels)

    if len(texts) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
        pipeline = _build_pipeline()
        pipeline.fit(X_train, y_train)
        accuracy = float(accuracy_score(y_test, pipeline.predict(X_test)))
    else:
        pipeline = _build_pipeline()
        pipeline.fit(texts, y)
        accuracy = 1.0

    joblib.dump(pipeline, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    logger.info("Modelo salvo. Acurácia: %.2f%%", accuracy * 100)
    return pipeline, encoder, accuracy


def predict(
    pipeline: Pipeline,
    encoder: LabelEncoder,
    texts: list[str],
) -> list[dict]:
    if not texts:
        return []
    proba_matrix = pipeline.predict_proba(texts)
    predictions = np.argmax(proba_matrix, axis=1)
    confidences = np.max(proba_matrix, axis=1)
    return [
        {
            "sentiment": encoder.inverse_transform([pred])[0],
            "confidence": round(float(conf), 4),
        }
        for pred, conf in zip(predictions, confidences)
    ]
