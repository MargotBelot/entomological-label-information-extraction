FROM python:3.10.14-bullseye
WORKDIR /app
RUN mkdir /app/label_processing
COPY ./label_processing/text_recognition.py /app/label_processing/text_recognition.py
COPY ./label_processing/utils.py /app/label_processing/utils.py
RUN mkdir -p /app/scripts/processing
COPY ./scripts/processing/tesseract.py /app/scripts/processing/tesseract.py
COPY ./pipelines/requirements/tesseract.txt /app/tesseract.txt
RUN apt-get update
RUN yes | apt-get install tesseract-ocr
RUN yes | apt install libtesseract-dev
RUN yes | apt-get install python3-opencv
RUN pip install -r tesseract.txt
# Set PYTHONPATH to find modules
ENV PYTHONPATH=/app
