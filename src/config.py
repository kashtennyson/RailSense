import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGE_SHAPE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# W&B settings
WANDB_PROJECT = "RailSense"
WANDB_ENTITY = 'kashtennyson'
USE_WANDB = True