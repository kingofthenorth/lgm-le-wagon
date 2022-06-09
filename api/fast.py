import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lgm_le_wagon.ml_logic.preprocessor import add_language
#from lgm_le_wagon.interface.main import predict_ooo
from lgm_le_wagon.interface.main_en import pred_en
from lgm_le_wagon.interface.mainfr import pred_fr


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
    return dict(greeting="hello Sam how are you?")


@app.get("/predict")
def predict(type,reply, first_message,_id):

    #reply = clean_text(reply)

    # if type == "GOOGLE_REPLY":
    #     X_pred = pd.DataFrame(dict(
    #         type= type,
    #         reply=reply,
    #         first_message=first_message,
    #         _id= _id
    #         ), index=[0])
    #     print(X_pred)
    #     y_pred,X_processed = predict_ooo(X_pred)
    #     print(y_pred)
    #     print(X_processed)

    #     if y_pred > 0.5:
    #         return {"Exception" : "Automatic reply detected, possibility of out of office"}

    langue = add_language(reply)
    if langue == "fr":
        X_pred_fr = pd.DataFrame(dict(
            type= type,
            reply=reply,
            first_message=first_message,
            _id= _id
            ), index=[0])

        y_pred_fr = pred_fr(X_pred_fr)
        return {"Sentiment analysis" : f'{y_pred_fr}'}
    elif langue == 'en':
        X_pred_en = pd.DataFrame(dict(
            type= type,
            reply=reply,
            first_message=first_message,
            _id= _id
            ), index=[0])

        y_pred_en = pred_en(X_pred_en)
        return {"Sentiment analysis" : f'{y_pred_en}'}
    else:
        return {"langue" : langue}


# X_pred = pd.DataFrame(dict(
#             type=['GOOGLE'],
#             reply=["I am not interested, don't contact me anymore"],
#             first_message=['Voici notre nouveau produit'],
#             _id=['xxxxxxxxxxxxxxxxxxxx']
#             ))





# http://127.0.0.1:8000/predict?type=GOOGLE_REPLY&reply=itisverycool&first_message=xxx&_id=zzz
# http://127.0.0.1:8000/predict?type=LINKEDIN_HAS_REPLY&reply=0001&first_message=0002&_id=0003
# http://127.0.0.1:8000/predict?type=0000&reply=0001&first_message=0002&_id=0003


    # # pipeline = get_model_from_gcp()
    # pipeline = joblib.load('model.joblib')  #vérifier correspondance nom du modele

    # # make prediction
    # results = pipeline.predict(X)

    # # convert response from numpy to python type
    # pred = float(results[0])

    # return dict(fare=pred)
