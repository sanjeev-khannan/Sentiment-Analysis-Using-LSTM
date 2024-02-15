import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Sentiment Analysis root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from config.paths import FilePathConstants
from config.model_configs import ModelConfigs
from models.LstmWithEmbedding import LSTMModel
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.saving import load_model
from utils.data_preprocessing import getSavedTokenizer, preprocessTexts
import argparse


class Prediction:
    def __init__(self) -> None:
        self.model = load_model(FilePathConstants.MODEL_SAVE_PATH)
        self.tokenizer = getSavedTokenizer()

    def __predict(self, texts):
        texts = preprocessTexts(texts, tokenizer_obj=self.tokenizer)
        result = self.model.predict(texts, verbose=0)
        return result

    def getPrediction(self, text: str):
        texts = pd.Series([text])
        result = self.__predict(texts)[0]
        neg, pos = result
        neg = "{:.2f}".format(neg)
        pos = "{:.2f}".format(pos)
        return neg, pos


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
    
    predictor = Prediction()
    neg, pos = predictor.getPrediction(text)

    print(f"Text -> {text}\n"
          f"Sentiment Probability -> Negative = {neg} | Positive = {pos}")


if __name__ == '__main__':
    main()
    