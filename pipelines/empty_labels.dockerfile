FROM python:3.10.14-bullseye
WORKDIR /app

ENV PYTHONPATH=/app

COPY . /app
COPY ./pipelines/requirements/empty_labels.txt /app/empty_labels.txt

RUN pip install -r empty_labels.txt
