import os
import pickle
from config.paths import FilePathConstants
from keras.saving import load_model


def getSavedTokenizer():
    if not os.path.isfile(FilePathConstants.TOKENIZER_SAVE_PATH):
        raise Exception(
            f"Saved tokenizer object not found in path - {FilePathConstants.TOKENIZER_SAVE_PATH}..\n")

    with open(FilePathConstants.TOKENIZER_SAVE_PATH, 'rb') as file:
        tokenizer_obj = pickle.load(file)

    return tokenizer_obj


def getSavedModel():
    if not os.path.isfile(FilePathConstants.MODEL_SAVE_PATH):
        raise Exception(
            f"Saved model object not found in path - {FilePathConstants.MODEL_SAVE_PATH}..\n")

    model = load_model(FilePathConstants.MODEL_SAVE_PATH)
    return model
