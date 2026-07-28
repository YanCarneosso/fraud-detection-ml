"""Executa o fluxo completo de detecção de fraudes."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from src.config import DATA_PATH, FIGURES_DIR, METRICS_DIR, MODELS_DIR, ROOT, Settings, ensure_directories
from src.data_processing import create_eda_figures, load_data, split_data, validate_data
from src.evaluate import calculate_metrics, save_evaluation_plots, threshold_analysis
from src.explain import create_shap_explanations
from src.report import generate_html_report
from src.train import build_candidates, choose_best, save_artifacts, train_and_compare, tune_model
from src.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento de detector de fraudes")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", choices=["all", "logistic", "random_forest", "xgboost"], default="all")
    parser.add_argument("--search", action="store_true", help="Otimiza o modelo vencedor no treino")
    parser.add_argument("--explain", action="store_true", help="Gera gráficos SHAP")
    parser.add_argument("--threshold", type=float, help="Threshold fixo; caso omitido, selecionado na validação")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--scoring", choices=["recall", "f1", "average_precision", "business_cost"],
                        default="average_precision")
    return parser.parse_args()


def main() -> int:
    """Orquestra EDA, treino, validação, teste final único e artefatos."""
    configure_logging()
    args = parse_args()
    ensure_directories()
    try:
        frame = load_data(args.data)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    diagnostics = validate_data(frame)
    LOGGER.info("Dados: %s", json.dumps(diagnostics, ensure_ascii=False, default=str))
    create_eda_figures(frame)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(frame, args.random_state)
    candidates = build_candidates(y_train, args.random_state)
    if args.model != "all":
        candidates = {k: v for k, v in candidates.items()
                      if k.startswith("dummy") or k.startswith(args.model)}
    fitted, comparison = train_and_compare(candidates, x_train, y_train, x_val, y_val)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    for name, model in fitted.items():
        val_probability = model.predict_proba(x_val)[:, 1]
        model_threshold = float(comparison.loc[
            comparison["Balanceamento"] == name, "Threshold"
        ].iloc[0])
        save_evaluation_plots(y_val, val_probability, threshold_analysis(y_val, val_probability),
                              f"validacao_{name}", FIGURES_DIR, model_threshold)
    best_name = choose_best(comparison)
    best_model = fitted[best_name]
    if args.search:
        LOGGER.info("Otimizando %s somente no conjunto de treino...", best_name)
        search = tune_model(best_model, x_train, y_train, args.scoring, args.random_state)
        best_model = search.best_estimator_
    val_probabilities = best_model.predict_proba(x_val)[:, 1]
    threshold_table = threshold_analysis(y_val, val_probabilities)
    threshold = args.threshold
    if threshold is None:
        threshold = float(comparison.loc[comparison["Balanceamento"] == best_name, "Threshold"].iloc[0])
    threshold_table.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)
    save_evaluation_plots(y_val, val_probabilities, threshold_table, "validacao", FIGURES_DIR, threshold)
    # Primeira e única consulta ao teste, depois de todas as escolhas.
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    final_metrics = calculate_metrics(y_test, test_probabilities, threshold)
    report = final_metrics.pop("classification_report")
    (METRICS_DIR / "classification_report_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "final_test_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_evaluation_plots(y_test, test_probabilities, threshold_table, "teste", FIGURES_DIR, threshold)
    save_artifacts(best_model, threshold, list(x_train.columns), final_metrics,
                   MODELS_DIR / "best_fraud_detection_pipeline.joblib",
                   MODELS_DIR / "model_metadata.json")
    if args.explain:
        try:
            create_shap_explanations(best_model, x_test, y_test, test_probabilities,
                                     threshold, FIGURES_DIR / "shap")
        except Exception:
            LOGGER.exception("Não foi possível gerar SHAP para o modelo escolhido.")
    report_path = generate_html_report(
        METRICS_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR.parent / "fraud_detection_report.html"
    )
    generate_html_report(METRICS_DIR, FIGURES_DIR, MODELS_DIR, ROOT / "docs" / "index.html")
    LOGGER.info("Relatório HTML: %s", report_path)
    LOGGER.info("Concluído. Modelo=%s; threshold=%.2f; métricas=%s",
                best_name, threshold, json.dumps(final_metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
