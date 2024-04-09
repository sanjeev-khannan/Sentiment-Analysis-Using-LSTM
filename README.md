# Sentiment Analysis using LSTM 🌐📊

Sentiment analysis on Amazon reviews using  Long Short-Term Memory (LSTM) model. The dataset, available on Kaggle, is preprocessed from raw text to train and test datasets. Model configurations, file paths, and training details are customizable. Training script checks for processed data, preprocesses if needed, and trains the LSTM model. Inference script predicts sentiment probabilities for user-provided text. 

## Dataset - Amazon Review dataset 
Download the dataset from Kaggle: [Amazon Review Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data)

The dataset is in raw format (.txt), each line starting with sentiment ('__label__1' or '__label__2') with a space and followed by the text.

Unpack the raw dataset to `data/raw/` which should have `train.ft.txt` and `test.ft.txt` files. During model training, if no processed data is found in the `data/processed/` folder, preprocessing utils will process the raw data and create train and test CSV files.

## Install Requirements ✅
It is recommendable to have python version <=3.11 <br>
Open the project root folder in terminal, and run the below command.
```bash
pip install -r requirements.txt
```

## Model Configs 🛠️
Model configuration parameters are located in `src/config/model_configs.py`. <br>
You can configure the model parameters, the default values are as follows,

VOCAB_SIZE = 20000

EMBEDDING_DIMENSION = 100

SENTENCE_MAX_LENGTH = 100

## File Paths 📂
File paths are specified in `src/config/paths.py`, which includes:

RAW_TRAIN_DATA_PATH <br>
RAW_TEST_DATA_PATH <br>
PROCESSED_TRAIN_DATA_PATH <br>
PROCESSED_TEST_DATA_PATH <br>
MODEL_SAVE_PATH <br>
TOKENIZER_SAVE_PATH <br>

## Model Training 🚀
To start the model training, navigate to `src/scripts/`

```bash
cd src/scripts/
```
```bash
python train.py
```
 - The script will automatically check for processed data; if not found, it will process and store it. 
 - If no tokenizer pickle object is found, it will fit a new tokenizer object and save it in the `checkpoints/` directory. 
 - By default, the model will train for 2 epochs, which can be modified in the `train.py` file. 
 - Once trained, the model will be stored in the `checkpoints/` folder.

## Model Object
You can download my trained model object from [here](https://drive.google.com/file/d/1JiYE1ypmlb3cj1A1W14m727y2WxGTIRc/view?usp=drive_link) <br>
Once downloaded, copy the `trained_model.keras` file to `checkpoints/` folder.

## Tokenizer Object
You can download my trained Tokenizer object from [here](https://drive.google.com/file/d/1XYGTFNAIRpx-GfUsDDfvGKlI8k5hr8HO/view?usp=sharing) <br>
Once downloaded, copy the `tokenizer.pickle` file to `checkpoints/` folder.

## Inference 🔍
To predict sentiment for a text, navigate to `src/scripts/`

```bash
cd src/scripts/
```
```bash
python predict.py --text "This is one of the best restaurants I have ever seen. Every dish is tasty."
```
Currently, you can pass a single text and get the results for class probabilities (Negative, Positive).

#### Sample Output 
```bash
Text -> This is one of the best restaurants I have ever seen. Every dish is tasty.
Sentiment Probability -> Negative = 0.01 | Positive = 0.99
```
