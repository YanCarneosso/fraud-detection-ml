"""Carregamento, validação, divisão e análise exploratória."""
from pathlib import Path
from typing import Any
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, REQUIRED_COLUMNS

LOGGER = logging.getLogger(__name__)


def load_data(path: Path | str) -> pd.DataFrame:
    """Carrega o CSV ou informa claramente como corrigir sua ausência."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{source}'. Baixe creditcard.csv, coloque-o "
            "em data/raw/creditcard.csv e execute novamente: python main.py"
        )
    frame = pd.read_csv(source)
    validate_data(frame)
    return frame


def validate_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Valida esquema, alvo, tipos e qualidade básica; retorna um diagnóstico."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
    unknown_target = set(frame["Class"].dropna().unique()) - {0, 1}
    if unknown_target:
        raise ValueError(f"Class contém valores inválidos: {sorted(unknown_target)}")
    non_numeric = [c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise TypeError(f"Colunas devem ser numéricas: {non_numeric}")
    counts = frame["Class"].value_counts().reindex([0, 1], fill_value=0)
    return {
        "total": int(len(frame)),
        "normais": int(counts[0]),
        "fraudes": int(counts[1]),
        "percentual_fraudes": float(counts[1] / len(frame) * 100) if len(frame) else 0.0,
        "proporcao_normal_fraude": float(counts[0] / counts[1]) if counts[1] else float("inf"),
        "ausentes": frame.isna().sum().to_dict(),
        "duplicados": int(frame.duplicated().sum()),
        "tipos": frame.dtypes.astype(str).to_dict(),
    }


def split_data(
    frame: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Divide em 70/15/15 com estratificação, mantendo teste isolado."""
    x, y = frame.drop(columns="Class"), frame["Class"].astype(int)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def create_eda_figures(frame: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Gera e salva as visualizações exploratórias solicitadas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    data["Amount_log"] = __import__("numpy").log1p(data["Amount"].clip(lower=0))
    specs = [
        ("distribuicao_classes.png", lambda ax: sns.countplot(data=data, x="Class", ax=ax),
         "Distribuição das classes", "Classe", "Quantidade"),
        ("distribuicao_amount.png", lambda ax: sns.histplot(data["Amount"], bins=60, ax=ax),
         "Distribuição dos valores", "Valor da transação", "Frequência"),
        ("distribuicao_amount_log.png", lambda ax: sns.histplot(data["Amount_log"], bins=60, ax=ax),
         "Distribuição logarítmica dos valores", "log(1 + valor)", "Frequência"),
        ("distribuicao_time.png", lambda ax: sns.histplot(data["Time"], bins=60, ax=ax),
         "Distribuição temporal", "Tempo (segundos)", "Frequência"),
        ("amount_por_classe.png", lambda ax: sns.boxplot(data=data, x="Class", y="Amount", ax=ax, showfliers=False),
         "Valores por classe", "Classe", "Valor da transação"),
    ]
    for filename, plotter, title, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotter(ax); ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)
    correlations = data.corr(numeric_only=True)["Class"].abs().nlargest(12).index
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data[correlations].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlação das variáveis mais relevantes")
    fig.tight_layout(); fig.savefig(output_dir / "matriz_correlacao.png", dpi=150); plt.close(fig)
    fraud = data.loc[data["Class"] == 1].assign(
        Hora=(data.loc[data["Class"] == 1, "Time"] // 3600).astype(int)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    fraud.groupby("Hora").size().plot(kind="bar", ax=ax)
    ax.set(title="Fraudes ao longo do tempo", xlabel="Hora transcorrida", ylabel="Quantidade de fraudes")
    fig.tight_layout(); fig.savefig(output_dir / "fraudes_tempo.png", dpi=150); plt.close(fig)
