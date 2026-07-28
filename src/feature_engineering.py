"""Transformações determinísticas disponíveis para treino e inferência."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def create_features(frame: pd.DataFrame, drop_originals: bool = True) -> pd.DataFrame:
    """Cria valor logarítmico e horário cíclico sem produzir infinitos."""
    required = {"Amount", "Time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Features ausentes: {sorted(missing)}")
    result = frame.copy()
    if (result["Amount"] < 0).any():
        raise ValueError("Amount não pode conter valores negativos")
    result["Amount_log"] = np.log1p(result["Amount"])
    result["Hour"] = (result["Time"] / 3600.0) % 24
    angle = 2 * np.pi * result["Hour"] / 24
    result["Hour_sin"] = np.sin(angle)
    result["Hour_cos"] = np.cos(angle)
    if drop_originals:
        result = result.drop(columns=["Amount", "Time", "Hour"])
    if not np.isfinite(result.select_dtypes(include=np.number).to_numpy()).all():
        raise ValueError("Features contêm valores não finitos")
    return result


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer sklearn para aplicar exatamente a mesma lógica em inferência."""

    def __init__(self, drop_originals: bool = True):
        self.drop_originals = drop_originals

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureEngineer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return create_features(x, self.drop_originals)

