import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Sentiment Analysis root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.data_preprocessing import preprocessTexts
from keras.saving import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
from models.LstmWithEmbedding import LSTMModel
from config.model_configs import ModelConfigs
from config.paths import FilePathConstants

def predict(texts):

    if not os.path.isfile(FilePathConstants.MODEL_SAVE_PATH):
        raise Exception(
            f"Saved model object not found in path - {FilePathConstants.MODEL_SAVE_PATH}..\n")

    texts = preprocessTexts(texts)
    model = load_model(FilePathConstants.MODEL_SAVE_PATH)
    result = model.predict(texts, verbose=0)
    del model
    return result


def main():
    parser = argparse.ArgumentParser(description='Text Sentiment Predictor')

    # Define command-line arguments
    parser.add_argument('--text', type=str, default='',
                        help="Pass a single text sentence. Ex: 'Hello World'")

    args = parser.parse_args()

    # Access the parsed arguments
    text = args.text

    if text == '':
        print(f"Input text argument is Invalid - '{text}'")
        return
    texts = pd.Series([text])
    result = predict(texts)[0]
    neg, pos = result
    neg = "{:.2f}".format(neg)
    pos = "{:.2f}".format(pos)

    print(f"Text -> {text}\n"
          f"Sentiment Probability -> Negative = {neg} | Positive = {pos}")


if __name__ == '__main__':
    main()
    