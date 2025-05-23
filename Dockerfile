# Dockerfile
FROM continuumio/miniconda3

WORKDIR /workspace
COPY environment.yml .

RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yml

SHELL ["conda", "run", "-n", "spatialhackathon", "/bin/bash", "-c"]
CMD ["python"]
