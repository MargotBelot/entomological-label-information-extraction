FROM python:3.10.14-bullseye
WORKDIR /app
RUN mkdir /app/label_processing
COPY ./label_processing/utils.py /app/label_processing/utils.py

RUN mkdir /app/label_postprocessing
COPY ./label_postprocessing/ocr_postprocessing.py /app/label_postprocessing/ocr_postprocessing.py
COPY ./scripts/postprocessing/process.py /app/process.py
COPY ./pipelines/requirements/postprocess.txt /app/postprocess.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r postprocess.txt
RUN python -c 'import nltk; nltk.download("punkt"); nltk.download("punkt_tab");'


