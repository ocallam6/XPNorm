FROM continuumio/miniconda3

RUN mkdir -p XPNorm

COPY . /XPNorm
WORKDIR /XPNorm

RUN conda env update --file environment.yml

RUN echo "conda activate XPNorm" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
