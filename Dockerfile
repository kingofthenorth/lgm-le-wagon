# syntax=docker/dockerfile:1
FROM python:3.8.13
COPY taxifare_api /taxifare_api
COPY taxifare_model /taxifare_model
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn taxifare_api.fast:app --host 0.0.0.0 --port $PORT
