"""Métricas, threshold, custo e gráficos de avaliação."""
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)


def business_cost(y_true: Any, y_pred: Any, fn_cost: float = 1000, fp_cost: float = 10) -> float:
    """Calcula custo hipotético configurável de falsos negativos e positivos."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * fn_cost + fp * fp_cost)


def apply_threshold(probabilities: Any, threshold: float) -> np.ndarray:
    """Converte probabilidades em classes."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold deve estar entre 0 e 1")
    return (np.asarray(probabilities) >= threshold).astype(int)


def calculate_metrics(y_true: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
    """Calcula métricas apropriadas usando probabilidades para métricas de ranking."""
    pred = apply_threshold(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "accuracy": accuracy_score(y_true, pred),
        "fraud_precision": precision_score(y_true, pred, zero_division=0),
        "fraud_recall": recall_score(y_true, pred, zero_division=0),
        "fraud_f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "false_positives": int(fp), "false_negatives": int(fn),
        "cost": business_cost(y_true, pred),
        "classification_report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }


def threshold_analysis(y_true: Any, probabilities: Any) -> pd.DataFrame:
    """Avalia thresholds na validação, incluindo obrigatoriamente 0,30."""
    thresholds = np.unique(np.append(np.arange(0.05, 0.95, 0.05), 0.30))
    return pd.DataFrame([calculate_metrics(y_true, probabilities, float(t)) for t in thresholds]).drop(
        columns="classification_report"
    )


def select_threshold(table: pd.DataFrame, criterion: str = "cost") -> float:
    """Seleciona threshold por menor custo ou maior métrica na validação."""
    if criterion == "cost":
        row = table.sort_values(["cost", "fraud_recall"], ascending=[True, False]).iloc[0]
    else:
        if criterion not in table:
            raise ValueError(f"Critério desconhecido: {criterion}")
        row = table.sort_values(criterion, ascending=False).iloc[0]
    return float(row["threshold"])


def save_evaluation_plots(y_true: Any, probabilities: Any, thresholds: pd.DataFrame,
                          prefix: str, output_dir: Path, threshold: float = 0.5) -> None:
    """Salva ROC, Precision-Recall, matriz de confusão e análise de threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    for name, x, y, xlabel, ylabel, title in [
        ("roc", fpr, tpr, "Taxa de falsos positivos", "Taxa de verdadeiros positivos", "Curva ROC"),
        ("pr", recall, precision, "Recall", "Precisão", "Curva Precision-Recall"),
    ]:
        fig, ax = plt.subplots(); ax.plot(x, y); ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_{name}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots()
    for column, label in [("fraud_precision", "Precisão"), ("fraud_recall", "Recall"), ("fraud_f1", "F1")]:
        ax.plot(thresholds["threshold"], thresholds[column], label=label)
    ax.set(xlabel="Threshold", ylabel="Métrica", title="Métricas por threshold"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_thresholds.png", dpi=150); plt.close(fig)
    matrix = confusion_matrix(y_true, apply_threshold(probabilities, threshold), labels=[0, 1])
    fig, ax = plt.subplots()
    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
    ax.set(title="Matriz de confusão", xlabel="Predição", ylabel="Classe real")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_matriz_confusao.png", dpi=150); plt.close(fig)
