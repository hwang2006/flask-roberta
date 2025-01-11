FROM python:3.11

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /webapp

RUN pip install -r requirements.txt

RUN python convert_to_onnx.py # create /webapp/oberta-sequence-classification-9.onnx

COPY webapp/* /webapp

#COPY roberta-sequence-classification-9.onnx /webapp

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
