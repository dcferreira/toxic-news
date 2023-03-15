FROM python:3.10 AS builder

# install hatch
RUN pip install --no-cache-dir --upgrade hatch
COPY . /code

# build python package
WORKDIR /code
RUN hatch build -t wheel

FROM python:3.10 as main

ENV TORCH_HOME /torch_home
RUN mkdir -p ${TORCH_HOME}/hub/checkpoints
RUN wget https://github.com/unitaryai/detoxify/releases/download/v0.4-alpha/multilingual_debiased-0b549669.ckpt \
    -O ${TORCH_HOME}/hub/checkpoints/multilingual_debiased-0b549669.ckpt

# copy wheel package from stage 1
COPY --from=builder /code/dist /code/dist
RUN pip install --no-cache-dir --upgrade /code/dist/*

# force download of model
RUN python -c "from detoxify import Detoxify; Detoxify('multilingual').predict(['foobar'])"

# copy the serving code and our database
COPY app.py /code

# run web server
WORKDIR /code
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
