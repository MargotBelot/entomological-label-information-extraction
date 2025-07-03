FROM python:3.10.14-bullseye
WORKDIR /app

ENV PYTHONPATH=/app

COPY . /app

RUN pip install pillow