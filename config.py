"""Default hyperparameters for CUT training.

Command-line options in train.py override these values for individual runs.
"""

DATA_ROOT = 'datasets/RRSHID'
OUTPUT_DIR = 'checkpoints/cut_baseline'

CROP_SIZE = 256
LOAD_SIZE = None
DISABLE_WEAK_DEGRADATION = False
BATCH_SIZE = 16
NUM_WORKERS = 8

N_EPOCHS_CONSTANT = 50
N_EPOCHS_COSINE = 50
NGF = 64
N_BLOCKS = 8
LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 0.0
LAMBDA_G = 0.5
LAMBDA_NCE = 2.0
PSNR_MAX_IMAGES = 0
PSNR_INTERVAL_EPOCHS = 5

DEVICE = 'cuda'
USE_AMP = True
SEED = 1407