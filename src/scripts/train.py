from pathlib import Path
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import tensorflow as tf
from models.LstmWithEmbedding import LSTMModel
from sklearn.metrics import classification_report
from utils.data_preprocessing import loadAndPreprocessData
from config.model_configs import ModelConfigs
from config.paths import FilePathConstants
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def train_model():

    print(f"Loading and Prepocessing Data\n")
    X_train, y_train, X_val, y_val, X_test, y_test = loadAndPreprocessData()

    print(f"Data Loaded... \n"
          f"TrainSet Size - {X_train.shape, y_train.shape}\n"
          f"ValidationSet Size - {X_val.shape, y_val.shape}\n"
          f"TestSet Size - {X_test.shape, y_test.shape}\n")

    model = LSTMModel(vocab_size=ModelConfigs.VOCAB_SIZE,
                      embedding_dim=ModelConfigs.EMBEDDING_DIMENSION,
                      max_length=ModelConfigs.SENTENCE_MAX_LENGTH)

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              epochs=2,
              validation_data=[
                  X_val,
                  y_val
              ])
    
    model.summary()
    
    model.save(FilePathConstants.MODEL_SAVE_PATH)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f"Evaluation Report\n"
          f"-----------------\n"
          f"Loss - {test_loss}\n"
          f"Accuracy - {test_accuracy}\n")

    y_true, y_pred = np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1)

    print(f"Classification Report\n"
          f"---------------------\n"
          f"{classification_report(y_true, y_pred)}")


if __name__ == '__main__':

    print(f"Sentiment Analyis model training on Amazon Review Dataset started...")

    train_model()

    print(f"Sentiment Analyis model training on Amazon Review Dataset completed successfully...")
