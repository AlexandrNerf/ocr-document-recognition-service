FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -n myenv python=3.12 && \
    echo "conda activate myenv" >> ~/.bashrc
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir -p assets/images

ENV HYDRA_FULL_ERROR=1

EXPOSE 8000

CMD ["python", "./app.py"]