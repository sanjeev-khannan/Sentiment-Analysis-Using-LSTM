from collections.abc import Iterable
import re
import os
import pandas as pd
from tensorflow import keras
from config.paths import FilePathConstants
from config.model_configs import ModelConfigs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import gc

from utils.objectLoader import getSavedTokenizer


def processData(file_path, new_file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fil:
            data = fil.readlines()

        for i in range(len(data)):
            data[i] = re.sub(r'__label__1', '0\t', data[i])
            data[i] = re.sub(r'__label__2', '1\t', data[i])
        data = ['sentiment\ttext\n'] + data

        with open(new_file_path, 'w') as fil:
            fil.writelines(data)

        del data
        gc.collect()
    else:
        raise Exception(f"Invalid file path for processing - {file_path}")


def parseSentence(sentence):
    try:
        sentence = re.sub(r'@[^ $]*', 'taggertoken', sentence.lower())
        sentence = re.sub(r'http:[a-zA-Z0-9./]*', 'urltoken', sentence)
        return re.sub(r"[^a-zA-Z0-9 ]", ' ', sentence)
    except Exception as e:
        print(sentence, e)
        raise (e)


def fitAndSaveTokenizer(text_sentences):

    print(
        f"Fitting Tokenier with with VocabSize - {ModelConfigs.VOCAB_SIZE}..\n")
    tokenizer = Tokenizer(num_words=ModelConfigs.VOCAB_SIZE)
    tokenizer.fit_on_texts(text_sentences)

    print(f"Saving Tokenzier object..")
    with open(FilePathConstants.TOKENIZER_SAVE_PATH, 'wb') as file:
        file.write(pickle.dumps(tokenizer))
    print(
        f"Tokenizer object saved at {FilePathConstants.TOKENIZER_SAVE_PATH}..\n")


def loadAndPreprocessData():

    if not os.path.isfile(FilePathConstants.PROCESSED_TRAIN_DATA_PATH):
        print("Processed train data not found.. cleaning raw train data..\n")
        processData(FilePathConstants.RAW_TRAIN_DATA_PATH,
                    FilePathConstants.PROCESSED_TRAIN_DATA_PATH)
    else:
        print("Processed train data found.. skipping train data processing..\n")

    if not os.path.isfile(FilePathConstants.PROCESSED_TEST_DATA_PATH):
        print("Processed test data not found.. cleaning raw test data..\n")
        processData(FilePathConstants.RAW_TEST_DATA_PATH,
                    FilePathConstants.PROCESSED_TEST_DATA_PATH)
    else:
        print("Processed test data found.. skipping test data processing..\n")

    print("Loading and splitting data..\n")
    train_data = pd.read_csv(
        FilePathConstants.PROCESSED_TRAIN_DATA_PATH, delimiter='\t')
    test_data = pd.read_csv(
        FilePathConstants.PROCESSED_TEST_DATA_PATH, delimiter='\t')

    X_train, X_val, y_train, y_val = train_test_split(train_data.text,
                                                      train_data.sentiment,
                                                      stratify=train_data.sentiment,
                                                      test_size=0.2,
                                                      random_state=42)

    X_test, y_test = test_data.text, test_data.sentiment
    print(f"Data Loaded succesfully..\n")

    X_train = X_train[:200]
    X_val = X_val[:200]
    X_test = X_test[:200]

    y_train = y_train[:200]
    y_val = y_val[:200]
    y_test = y_test[:200]

    print("Processing text inputs..\n")
    X_train = X_train.apply(lambda x: parseSentence(x))
    X_val = X_val.apply(lambda x: parseSentence(x))
    X_test = X_test.apply(lambda x: parseSentence(x))

    print("Processing target inputs..\n")
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    y_test = keras.utils.to_categorical(y_test)

    if not os.path.isfile(FilePathConstants.TOKENIZER_SAVE_PATH):
        print(
            f"Saved tokenizer object not found in path - {FilePathConstants.TOKENIZER_SAVE_PATH}..\n")
        fitAndSaveTokenizer(X_train)

    print(
        f"Loading tokenizer object from {FilePathConstants.TOKENIZER_SAVE_PATH}\n")
    with open(FilePathConstants.TOKENIZER_SAVE_PATH, 'rb') as file:
        tokenizer = pickle.load(file)

    print(f"Sequencing and padding text inputs..\n")
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train,
                            maxlen=ModelConfigs.SENTENCE_MAX_LENGTH,
                            padding='post')
    X_val = pad_sequences(X_val,
                          maxlen=ModelConfigs.SENTENCE_MAX_LENGTH,
                          padding='post')
    X_test = pad_sequences(X_test,
                           maxlen=ModelConfigs.SENTENCE_MAX_LENGTH,
                           padding='post')

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocessTexts(texts, tokenizer_obj=None):
    if not isinstance(texts, Iterable):
        raise Exception(
            "'texts' input is not iterable. Pass texts as an iterable object")

    if tokenizer_obj == None:
        tokenizer_obj = getSavedTokenizer()

    texts = texts.apply(lambda x: parseSentence(x))
    texts = tokenizer_obj.texts_to_sequences(texts)
    texts = pad_sequences(texts,
                          maxlen=ModelConfigs.SENTENCE_MAX_LENGTH,
                          padding='post')

    return texts

