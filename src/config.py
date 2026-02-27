# ================================
# Project Configuration File
# ================================

import os

# -------------------------------
# Paths
# -------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"

# -------------------------------
# PCA Parameters
# -------------------------------

PCA_COMPONENTS = 5

# -------------------------------
# HMM Parameters
# -------------------------------

N_HIDDEN_STATES = 5
COVARIANCE_TYPE = "full"
N_ITER = 100
TOLERANCE = 1e-4
RANDOM_STATE = 42

# -------------------------------
# RUL Parameters
# -------------------------------

MAX_RUL_CAP = 125   # Common practice in NASA dataset

