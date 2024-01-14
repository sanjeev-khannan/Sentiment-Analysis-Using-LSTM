import os
class FilePathConstants:
    RAW_TRAIN_DATA_PATH = os.path.abspath("../../data/raw/train.ft.txt")
    RAW_TEST_DATA_PATH = os.path.abspath("../../data/raw/train.ft.txt")

    PROCESSED_TRAIN_DATA_PATH = os.path.abspath("../../data/processed/processed_train_data.csv")
    PROCESSED_TEST_DATA_PATH = os.path.abspath("../../data/processed/processed_test_data.csv")

    MODEL_SAVE_PATH = os.path.abspath("../../checkpoints/trained_model.keras")

    TOKENIZER_SAVE_PATH = os.path.abspath("../../checkpoints/tokenizer.pickle")