FROM python:3.8.6-buster

COPY api /api
COPY lgm-le-wagon /lgm-le-wagon
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
#COPY /Users/boudraasami/code/Samibou23/esoteric-virtue-346915-6b5b52544e75.json /credentials.json

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
