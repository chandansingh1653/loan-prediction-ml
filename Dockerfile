FROM  ubuntu:18.04
RUN apt-get update
RUN apt-get install -y python python-dev python-pip python-virtualenv
RUN apt-get install -y python3-pip git
ADD requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
RUN mkdir -p /data
COPY loan_predictions/  /data/loan_predictions
COPY model.py data/
WORKDIR /data
CMD python3 model.py
