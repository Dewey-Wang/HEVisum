# Dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /workspace
COPY environment.yml .

RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yml && \
    conda clean -afy 

SHELL ["conda", "run", "-n", "spatialhackathon", "/bin/bash", "-c"]
CMD ["python"]

