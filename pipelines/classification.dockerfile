FROM python:3.9-bullseye
WORKDIR /app
RUN mkdir /app/label_processing
COPY ./label_processing/utils.py /app/label_processing/utils.py

COPY ./label_processing/tensorflow_classifier.py /app/label_processing/tensorflow_classifier.py
RUN mkdir -p /app/scripts/processing
COPY ./models/classes /app/classes
COPY ./models/label_classifier_hp /models/label_classifier_hp
COPY ./models/label_classifier_identifier_not_identifier /models/label_classifier_identifier_not_identifier
COPY ./models/label_classifier_multi_single /app/label_classifier_multi_single
COPY ./scripts/processing/classifiers.py /app/classifiers.py
COPY ./pipelines/requirements/classifier.txt /app/classifier.txt
RUN apt-get update
RUN yes | apt-get install python3-opencv
RUN pip install -r classifier.txt
