FROM python:alpine
RUN apk update
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN mkdir -p /data
COPY loan_predictions/  /data/loan_predictions
COPY model.py data/
WORKDIR /data
CMD python model.py
