import numpy as np
import pandas as pd

from src.modeling import category_levels, load_bundle, save_bundle, to_lgbm_frame


def test_category_codes_match_across_samples_and_unseen_becomes_nan():
    train = pd.DataFrame({"cat": ["b", "a", "c", None], "num": [1.0, 2.0, 3.0, 4.0]})
    valid = pd.DataFrame({"cat": ["c", "z", "a"], "num": [5.0, 6.0, np.nan]})
    levels = category_levels(train, ["cat"])
    assert levels == {"cat": ["a", "b", "c"]}

    X_train = to_lgbm_frame(train, ["cat", "num"], levels)
    X_valid = to_lgbm_frame(valid, ["cat", "num"], levels)
    assert list(X_train["cat"].cat.categories) == list(X_valid["cat"].cat.categories)
    assert X_valid["cat"].cat.codes.tolist() == [2, -1, 0]   # "z" unseen in train -> NaN (code -1)
    assert X_valid["num"].isna().sum() == 1                    # numeric columns untouched


def test_bundle_round_trip(tmp_path):
    path = save_bundle(tmp_path / "models" / "m.joblib", features=["a", "b"], threshold=0.1)
    bundle = load_bundle(path)
    assert bundle == {"features": ["a", "b"], "threshold": 0.1}
