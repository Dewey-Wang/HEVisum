# Dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /workspace

# Install system build tools required for gmpy2, psutil, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN conda update -n base -c defaults conda -y && \
    conda install -n base -c conda-forge \
        pip jupyterlab numpy pandas matplotlib && \
    conda create -n spatialhackathon python=3.9 -y && \
    conda run -n spatialhackathon pip install -r requirements.txt && \
    conda run -n spatialhackathon python -m ipykernel install --user --name=spatialhackathon --display-name "Python (spatialhackathon)" && \
    conda clean -afy
