"""Interface segura de inferência."""
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd


def predict_transaction(transaction: dict[str, Any], model_path: Path | str,
                        metadata_path: Path | str) -> dict[str, float | str]:
    """Valida uma transação e retorna probabilidade, threshold e classificação."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    expected = metadata["raw_features"]
    missing, unknown = sorted(set(expected) - set(transaction)), sorted(set(transaction) - set(expected))
    if missing or unknown:
        raise ValueError(f"Features ausentes: {missing}; desconhecidas: {unknown}")
    try:
        frame = pd.DataFrame([[transaction[c] for c in expected]], columns=expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Todos os valores devem ser numéricos") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Valores devem ser finitos")
    model = joblib.load(model_path)
    probability = float(model.predict_proba(frame)[:, 1][0])
    threshold = float(metadata["threshold"])
    return {"probabilidade_fraude": probability, "threshold": threshold,
            "classificacao": "fraude" if probability >= threshold else "normal"}

