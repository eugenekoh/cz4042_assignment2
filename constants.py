"""
Constants used in PartB.
"""

from pathlib import Path

# Model
BATCH_SIZE = 128
MAX_DOCUMENT_LENGTH = 100
EPOCHS = 20
LR = 0.01
DROP_RATE = 0.4
CLIP_VALUE = 2
SEED = 10

# Data location
TRAIN_FILE = './data/train_medium.csv'
TEST_FILE = './data/test_medium.csv'

# Graphing
IMG_PATH = Path('results/graphs/partB')
