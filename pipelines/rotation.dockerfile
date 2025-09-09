FROM python:3.10.14-bullseye
WORKDIR /app
RUN mkdir /app/label_processing
COPY ./label_processing/utils.py /app/label_processing/utils.py

COPY ./label_processing/label_rotation.py /app/label_processing/label_rotation.py
RUN mkdir -p /app/scripts/processing
COPY ./scripts/processing/rotation.py /app/scripts/processing/rotation.py
COPY ./pipelines/requirements/rotation.txt /app/rotation.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r rotation.txt
RUN mkdir -p /app/models
COPY models/rotation_model.h5 /app/models/rotation_model.h5
# Set PYTHONPATH to find modules
ENV PYTHONPATH=/app
