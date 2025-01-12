# FROM python:3.11

# Use the official Python 3.11 image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and to enable buffering for logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /webapp

#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY convert_to_onnx.py /webapp

# create roberta-sequence-classification-9.onnx
RUN python convert_to_onnx.py 

COPY webapp/* /webapp

# COPY roberta-sequence-classification-9.onnx /webapp

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
