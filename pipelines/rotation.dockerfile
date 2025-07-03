FROM python:3.10.14-bullseye
WORKDIR /app
RUN mkdir /app/label_processing
COPY ./label_processing/utils.py /app/label_processing/utils.py

COPY ./label_processing/label_rotation.py /app/label_processing/label_rotation.py
COPY ./scripts/processing/rotation.py /app/rotation.py
COPY ./pipelines/requirements/rotation.txt /app/rotation.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r rotation.txt
COPY models/rotation_model.h5 /models/rotation_model.h5