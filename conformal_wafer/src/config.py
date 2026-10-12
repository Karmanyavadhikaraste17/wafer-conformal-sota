import os
import torch

class Config:
    # Random seed for full reproducibility
    SEED = 42

    # Project directories
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DATA_PATH = os.path.join(DATA_DIR, "Wafer_Map_Datasets.npz")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    CLASS_MAP_PATH = os.path.join(RESULTS_DIR, "class_map.csv")

    # Dataset properties
    NUM_CLASSES = 38
    IMG_SIZE = 52
    IN_CHANNELS = 3
    
    # Stratified split proportions
    # 60% Train, 20% Calibration, 20% Test
    # Validation (for early stopping & LR schedule) is 10% carved from Train split
    TRAIN_RATIO = 0.60
    CAL_RATIO = 0.20
    TEST_RATIO = 0.20
    VAL_FROM_TRAIN_RATIO = 0.10

    # Hardware & Training setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = torch.cuda.is_available()
    BATCH_SIZE = 128
    NUM_WORKERS = 0  # 0 for Windows safety
    EPOCHS = 35
    PATIENCE = 8
    LR = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4

    # Conformal Prediction settings
    ALPHAS = [0.01, 0.05, 0.10]
    PRIMARY_ALPHA = 0.10  # 90% target coverage
    RAPS_LAMBDA = 0.01    # Regularization penalty weight
    RAPS_K_REG = 1        # Set size penalty index

    # Robustness Study
    NOISE_RATES = [0.05, 0.10, 0.20]

    # Ablation study
    LABEL_SMOOTHING_VALUES = [0.0, 0.1]
    K_REG_SWEEP = [0.5, 1.0, 2.0]

# Instantiate default config singleton
cfg = Config()
