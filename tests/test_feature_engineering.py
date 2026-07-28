import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import create_features


def test_amount_log_and_time_features():
    result = create_features(pd.DataFrame({"Amount": [0.0, 9.0], "Time": [0.0, 21600.0]}),
                             drop_originals=False)
    assert result["Amount_log"].tolist() == pytest.approx([0, np.log(10)])
    assert {"Hour", "Hour_sin", "Hour_cos"} <= set(result.columns)


def test_no_infinite_values():
    result = create_features(pd.DataFrame({"Amount": [1e300], "Time": [1e10]}))
    assert np.isfinite(result.to_numpy()).all()


def test_negative_amount_is_rejected():
    with pytest.raises(ValueError, match="negativos"):
        create_features(pd.DataFrame({"Amount": [-1], "Time": [0]}))

