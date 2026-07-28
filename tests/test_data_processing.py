import pandas as pd
import pytest

from src.config import REQUIRED_COLUMNS
from src.data_processing import load_data, validate_data


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame([[0.0] * 30 + [0]], columns=REQUIRED_COLUMNS)


def test_load_data(tmp_path):
    path = tmp_path / "creditcard.csv"
    valid_frame().to_csv(path, index=False)
    assert load_data(path).shape == (1, 31)


def test_load_missing_file_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw/creditcard.csv"):
        load_data(tmp_path / "missing.csv")


def test_required_columns():
    with pytest.raises(ValueError, match="Colunas obrigatórias"):
        validate_data(valid_frame().drop(columns="V28"))


def test_duplicate_detection():
    diagnostics = validate_data(pd.concat([valid_frame(), valid_frame()], ignore_index=True))
    assert diagnostics["duplicados"] == 1

