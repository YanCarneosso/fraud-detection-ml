import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.evaluate import apply_threshold, business_cost
from src.inference import predict_transaction


def test_threshold_and_cost():
    pred = apply_threshold([0.2, 0.3, 0.9], 0.3)
    assert pred.tolist() == [0, 1, 1]
    assert business_cost([1, 0, 1], pred, fn_cost=1000, fp_cost=10) == 1010


def test_inference_response(tmp_path):
    model = Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(
        pd.DataFrame({"a": [0.0, 1.0]}), [0, 1]
    )
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps({"threshold": 0.3, "raw_features": ["a"]}), encoding="utf-8")
    response = predict_transaction({"a": 2}, model_path, metadata_path)
    assert set(response) == {"probabilidade_fraude", "threshold", "classificacao"}
    assert response["classificacao"] in {"normal", "fraude"}


def test_inference_rejects_unknown_feature(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"threshold": 0.5, "raw_features": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="desconhecidas"):
        predict_transaction({"a": 1, "b": 2}, tmp_path / "unused", metadata)

