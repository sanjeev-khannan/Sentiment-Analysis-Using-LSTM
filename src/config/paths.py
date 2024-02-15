import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[2])  # Sentiment Analysis root directory


class FilePathConstants:
    RAW_TRAIN_DATA_PATH = os.path.abspath(ROOT+"/data/raw/train.ft.txt")
    RAW_TEST_DATA_PATH = os.path.abspath(ROOT+"/data/raw/train.ft.txt")

    PROCESSED_TRAIN_DATA_PATH = os.path.abspath(
        ROOT+"/data/processed/processed_train_data.csv")
    PROCESSED_TEST_DATA_PATH = os.path.abspath(
        ROOT+"/data/processed/processed_test_data.csv")

    MODEL_SAVE_PATH = os.path.abspath(ROOT+"/checkpoints/trained_model.keras")

    TOKENIZER_SAVE_PATH = os.path.abspath(ROOT+"/checkpoints/tokenizer.pickle")
