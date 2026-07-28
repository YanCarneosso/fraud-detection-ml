"""Configurações centrais do projeto."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"
REQUIRED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


@dataclass(frozen=True)
class Settings:
    """Parâmetros reproduzíveis e custos hipotéticos."""

    random_state: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    false_negative_cost: float = 1000.0
    false_positive_cost: float = 10.0
    scoring: str = "average_precision"
    cv_splits: int = 5


def ensure_directories() -> None:
    """Cria diretórios de saída."""
    for path in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, FIGURES_DIR / "shap"):
        path.mkdir(parents=True, exist_ok=True)

