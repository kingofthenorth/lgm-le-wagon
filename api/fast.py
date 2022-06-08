# $DELETE_BEGIN
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.helper import *

from lgm_le_wagon.ml_logic.preprocessor import clean_text, add_language



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#url = 'http://127.0.0.1:8000/predict'


@app.get("/")
def index():
    return dict(greeting="hello")

@app.get("/predict")
def predict(type,
            reply,
            first_message,
            _id):

    if type == "GOOGLE_REPLY":   #GOOGLE_REPLY = mail
        return dict(
        type=['email detected'],
        reply=[float(reply)],
        _id=[float(_id)])



    elif type == "LINKEDIN_HAS_REPLY":  #LINKEDIN_HAS_REPLY=linkedin
        return dict(
        type=['linkedin message detected'],
        reply=[float(reply)],
        _id=[float(_id)])

    else:
        return dict(
        type=['type not detected'],
        reply=[float(reply)],
        _id=[float(_id)])





@app.get("/predict")
def predict(type,reply,_id):
    cleaned_reply = clean_text(reply)
    if type == "mail":






# http://127.0.0.1:8000/predict?type=GOOGLE_REPLY&reply=0001&first_message=0002&_id=0003
# http://127.0.0.1:8000/predict?type=LINKEDIN_HAS_REPLY&reply=0001&first_message=0002&_id=0003
# http://127.0.0.1:8000/predict?type=0000&reply=0001&first_message=0002&_id=0003


    # # pipeline = get_model_from_gcp()
    # pipeline = joblib.load('model.joblib')  #vérifier correspondance nom du modele

    # # make prediction
    # results = pipeline.predict(X)

    # # convert response from numpy to python type
    # pred = float(results[0])

    # return dict(fare=pred)
