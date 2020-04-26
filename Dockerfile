FROM python:alpine
RUN apk update

RUN apk add --no-cache python3-dev libstdc++ && \
    apk add --no-cache g++ && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN apk --no-cache add lapack libstdc++ && \
    apk add --no-cache python3-dev libstdc++ && \
    apk --no-cache add --virtual .builddeps g++ gcc gfortran musl-dev lapack-dev && \
    apk del .builddeps && \
    rm -rf /root/.cache

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN mkdir -p /data
COPY loan_predictions/  /data/loan_predictions
COPY model.py data/
WORKDIR /data
CMD python model.py
