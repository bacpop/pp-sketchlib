FROM python:3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                          cmake \
                          gfortran \
                          libeigen3-dev \
                          libopenblas-dev

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

COPY . /src
WORKDIR /src
RUN python setup.py install
