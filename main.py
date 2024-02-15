from fastapi import FastAPI

from src.scripts.predict import Prediction

app = FastAPI(title="SentimentPredictor")

predictor = Prediction()

@app.get("/")
async def homePage():
    return "<h2>Sentiment Predictor</h2> <br> Please pass text as query parameter!!"

@app.post("/predict/")
async def getPrediction(text:str):
    neg, pos = predictor.getPrediction(text)
    response_text = {
        'text': text,
        'scores': {
            'positive' : pos,
            'negative': neg
        }
    }
    return response_text