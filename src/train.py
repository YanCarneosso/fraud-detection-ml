"""Construção, comparação, otimização e persistência de modelos."""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if __package__:
    from .evaluate import (
        business_cost,
        calculate_metrics,
        select_threshold,
        threshold_analysis,
    )
    from .feature_engineering import FraudFeatureEngineer
else:
    # Permite que IDEs carreguem este arquivo diretamente, embora o fluxo deva
    # ser iniciado por main.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluate import (
        business_cost,
        calculate_metrics,
        select_threshold,
        threshold_analysis,
    )
    from src.feature_engineering import FraudFeatureEngineer

LOGGER = logging.getLogger(__name__)


def negative_business_cost(y_true: Any, y_pred: Any) -> float:
    """Scorer: maior é melhor, portanto retorna o custo com sinal negativo."""
    return -business_cost(y_true, y_pred)


def build_candidates(y_train: pd.Series, random_state: int = 42) -> dict[str, Pipeline]:
    """Cria baseline e combinações de modelo/balanceamento sem ajustar dados."""
    feature = FraudFeatureEngineer()
    logistic = LogisticRegression(max_iter=2000, random_state=random_state)
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates: dict[str, Pipeline] = {
        "dummy_original": Pipeline([("features", feature), ("model", DummyClassifier(strategy="prior"))]),
        "logistic_original": Pipeline([("features", feature), ("scale", StandardScaler()), ("model", logistic)]),
        "logistic_class_weight": Pipeline([("features", feature), ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))]),
        "logistic_under": Pipeline([("features", feature), ("scale", StandardScaler()),
            ("balance", RandomUnderSampler(random_state=random_state)), ("model", clone(logistic))]),
        "logistic_smote": Pipeline([("features", feature), ("scale", StandardScaler()),
            ("balance", SMOTE(random_state=random_state)), ("model", clone(logistic))]),
        "random_forest_class_weight": Pipeline([("features", feature), ("model", RandomForestClassifier(
            n_estimators=250, max_depth=10, class_weight="balanced", n_jobs=-1,
            random_state=random_state))]),
        "xgboost_weighted": Pipeline([("features", feature), ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=ratio, eval_metric="logloss",
            n_jobs=-1, random_state=random_state))]),
    }
    return candidates


def tune_model(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series,
               scoring: str = "average_precision", random_state: int = 42) -> Any:
    """Otimiza hiperparâmetros apenas no treino com CV estratificada."""
    model = pipeline.named_steps["model"]
    if isinstance(model, RandomForestClassifier):
        params = {"model__n_estimators": [150, 250, 400], "model__max_depth": [8, 10, 16, None],
                  "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4],
                  "model__max_features": ["sqrt", "log2"]}
    elif isinstance(model, XGBClassifier):
        params = {"model__n_estimators": [200, 300, 500], "model__max_depth": [3, 5, 7],
                  "model__learning_rate": [0.02, 0.05, 0.1], "model__subsample": [0.7, 0.9, 1.0],
                  "model__colsample_bytree": [0.7, 0.9, 1.0], "model__min_child_weight": [1, 5],
                  "model__reg_alpha": [0, 0.1, 1], "model__reg_lambda": [1, 5, 10]}
    else:
        params = {"model__C": np.logspace(-3, 2, 10)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    if scoring == "business_cost":
        scorer: Any = make_scorer(negative_business_cost)
    else:
        scorer = scoring
    search = RandomizedSearchCV(pipeline, params, n_iter=min(20, np.prod([len(v) for v in params.values()])),
                                scoring=scorer, cv=cv, n_jobs=-1, random_state=random_state, refit=True)
    return search.fit(x_train, y_train)


def train_and_compare(candidates: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series,
                      x_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Treina no treino e seleciona thresholds exclusivamente na validação."""
    fitted, rows = {}, []
    for name, pipeline in candidates.items():
        started = perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = perf_counter() - started
        probabilities = pipeline.predict_proba(x_val)[:, 1]
        table = threshold_analysis(y_val, probabilities)
        threshold = 0.5 if name.startswith("dummy") else select_threshold(table, "cost")
        metrics = calculate_metrics(y_val, probabilities, threshold)
        rows.append({"Modelo": name.split("_")[0], "Balanceamento": name,
                     "Threshold": threshold, "Precisão da fraude": metrics["fraud_precision"],
                     "Recall da fraude": metrics["fraud_recall"], "F1 da fraude": metrics["fraud_f1"],
                     "ROC-AUC": metrics["roc_auc"], "PR-AUC": metrics["average_precision"],
                     "Falsos positivos": metrics["false_positives"], "Falsos negativos": metrics["false_negatives"],
                     "Custo": metrics["cost"], "Tempo de treinamento": elapsed})
        fitted[name] = pipeline
    return fitted, pd.DataFrame(rows)


def choose_best(comparison: pd.DataFrame) -> str:
    """Escolhe por custo, desempata por PR-AUC e recall; não usa ROC-AUC isoladamente."""
    eligible = comparison[~comparison["Balanceamento"].str.startswith("dummy")]
    return str(eligible.sort_values(["Custo", "PR-AUC", "Recall da fraude"],
                                    ascending=[True, False, False]).iloc[0]["Balanceamento"])


def save_artifacts(model: Any, threshold: float, raw_features: list[str], metrics: dict[str, Any],
                   model_path: Path, metadata_path: Path) -> None:
    """Persiste pipeline completo e metadados necessários à inferência."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {"threshold": threshold, "raw_features": raw_features,
                "model_class": type(model.named_steps["model"]).__name__, "metrics": metrics}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
