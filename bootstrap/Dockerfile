FROM continuumio/miniconda3:latest
RUN python -V
RUN apt update -qq \
    && apt install -yqq gfortran mpi-default-dev \
    && rm -rf /var/lib/apt/lists/*
RUN conda install -yc conda-forge galario
WORKDIR /diskchef
