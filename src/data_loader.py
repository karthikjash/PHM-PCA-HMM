# =====================================
# Data Loader for NASA C-MAPSS Dataset
# =====================================

import os
import pandas as pd
from .config import DATA_RAW_PATH, TRAIN_FILE, TEST_FILE, RUL_FILE


def _get_column_names():
    """
    Returns standard column names for C-MAPSS dataset.
    """
    columns = ["unit_number", "time_in_cycles"]

    # 3 operational settings
    columns += [f"op_setting_{i}" for i in range(1, 4)]

    # 21 sensor measurements
    columns += [f"sensor_{i}" for i in range(1, 22)]

    return columns


def load_training_data():
    """
    Loads training dataset.
    """
    file_path = os.path.join(DATA_RAW_PATH, TRAIN_FILE)

    columns = _get_column_names()
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = columns

    return df


def load_test_data():
    """
    Loads test dataset.
    """
    file_path = os.path.join(DATA_RAW_PATH, TEST_FILE)

    columns = _get_column_names()
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = columns

    return df


def load_rul_data():
    """
    Loads true RUL values for test dataset.
    """
    file_path = os.path.join(DATA_RAW_PATH, RUL_FILE)

    rul_df = pd.read_csv(file_path, sep=r"\s+", header=None)
    rul_df.columns = ["RUL"]

    return rul_df

