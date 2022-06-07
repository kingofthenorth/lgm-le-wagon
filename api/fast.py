# $DELETE_BEGIN
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/")
def index():
    return dict(greeting="hello")

@app.get("/predict")
def predict(log_type,
            reply,
            first_message,
            _id):

    return dict(
        log_type=[log_type],
        reply=[float(reply)],
        first_message=[float(first_message)],
        _id=[float(_id)])

    X = pd.DataFrame(dict(       # vérifier correspondance clées
        key=[key],
        log_type=[log_type],
        reply=[float(reply)],
        first_message=[float(first_message)],
        _id=[float(_id)]))

# http://127.0.0.1:8000/predict?log_type=0000&reply=0001&first_message=0002&_id=0003

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')  #vérifier correspondance nom du modele

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(fare=pred)
