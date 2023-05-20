FROM python:3.10 AS builder

# install hatch
RUN pip install --no-cache-dir --upgrade hatch
COPY . /code

# build python package
WORKDIR /code
RUN hatch build -t wheel

FROM python:3.10 as main

# copy wheel package from stage 1
COPY --from=builder /code/dist /code/dist
RUN pip install --no-cache-dir --upgrade /code/dist/*

# force download of models
RUN python -c 'from optimum.onnxruntime import ORTModelForSequenceClassification;\
    model = ORTModelForSequenceClassification.from_pretrained("dcferreira/detoxify-optimized")'
RUN python -c 'from transformers import pipeline;\
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment";\
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)'

# copy the serving code and our database
COPY app.py /code

# run web server
WORKDIR /code
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
