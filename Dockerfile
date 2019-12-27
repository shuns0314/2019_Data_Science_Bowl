FROM python:3.7.6-buster

WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt