import pandas as pd
from pathlib import Path

def test_feature_files_exist():
    proc = Path('data/processed')
    assert (proc/'X_train.csv').exists(), "X_train.csv missing"
    assert (proc/'y_train.csv').exists(), "y_train.csv missing"