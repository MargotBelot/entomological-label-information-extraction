FROM python:3.10.14-bullseye
WORKDIR /app
# Copy the entire label_processing package
COPY ./label_processing /app/label_processing
RUN mkdir -p /app/scripts/processing
COPY ./models/classes /app/classes
RUN mkdir -p /app/label_processing/models
COPY ./models/label_classifier_hp /app/label_processing/models/label_classifier_hp
COPY ./models/label_classifier_identifier_not_identifier /app/label_processing/models/label_classifier_identifier_not_identifier
COPY ./models/label_classifier_multi_single /app/label_processing/models/label_classifier_multi_single
RUN mkdir -p /app/scripts/processing
COPY ./scripts/processing/classifiers.py /app/scripts/processing/classifiers.py
COPY ./pipelines/requirements/classifier.txt /app/classifier.txt
RUN apt-get update
RUN yes | apt-get install python3-opencv
RUN pip install -r classifier.txt
# Ensure Python can import from /app
ENV PYTHONPATH=/app
