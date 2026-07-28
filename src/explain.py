"""Explicabilidade SHAP para modelos compatíveis."""
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

LOGGER = logging.getLogger(__name__)


def _positive_values(explanation: shap.Explanation) -> shap.Explanation:
    """Normaliza formatos SHAP binários entre versões."""
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        return explanation[:, :, 1]
    return explanation


def create_shap_explanations(pipeline: object, x: pd.DataFrame, y: pd.Series,
                             probabilities: np.ndarray, threshold: float, output_dir: Path) -> None:
    """Gera explicações globais e locais; SHAP indica contribuição, não causalidade."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = pipeline[:-1].transform(x)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(transformed)
    sample = transformed.sample(min(1000, len(transformed)), random_state=42)
    model = pipeline.named_steps["model"]
    explanation = _positive_values(shap.Explainer(model, sample)(sample))
    for kind, filename in [("dot", "summary_plot.png"), ("bar", "bar_plot.png")]:
        shap.summary_plot(explanation.values, sample, plot_type=kind, show=False)
        plt.tight_layout(); plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight"); plt.close()
    pred = (probabilities >= threshold).astype(int)
    cases = {
        "fraude_detectada": np.flatnonzero((np.asarray(y) == 1) & (pred == 1)),
        "falso_positivo": np.flatnonzero((np.asarray(y) == 0) & (pred == 1)),
        "falso_negativo": np.flatnonzero((np.asarray(y) == 1) & (pred == 0)),
    }
    full_explainer = shap.Explainer(model, sample)
    for label, indices in cases.items():
        if len(indices):
            row = transformed.iloc[[indices[0]]]
            local = _positive_values(full_explainer(row))
            shap.plots.waterfall(local[0], show=False)
            plt.tight_layout(); plt.savefig(output_dir / f"{label}.png", dpi=150, bbox_inches="tight"); plt.close()
        else:
            LOGGER.info("Caso SHAP '%s' inexistente na amostra de teste.", label)

