"""Utilitários compartilhados."""
import logging


def configure_logging() -> None:
    """Configura logs legíveis para linha de comando."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # Matplotlib registra conversões internas de eixos categóricos como INFO.
    # Elas não representam problemas nos dados e poluem a saída do treinamento.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
