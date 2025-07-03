FROM python:3.10.14-bullseye
WORKDIR /app
RUN mkdir /app/label_processing
COPY ./label_processing/utils.py /app/label_processing/utils.py