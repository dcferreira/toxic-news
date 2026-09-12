FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/

# Install the project environment straight into the system prefix, so no
# separate venv has to be copied between build stages.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /code

# Locked dependencies first: this layer only invalidates when the lock changes.
COPY pyproject.toml uv.lock README.md LICENSE.txt ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# force download of models, so the container never fetches them at request time
RUN python -c 'from optimum.onnxruntime import ORTModelForSequenceClassification;\
    model = ORTModelForSequenceClassification.from_pretrained("dcferreira/detoxify-optimized")'
RUN python -c 'from transformers import pipeline;\
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment";\
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)'

# The pipeline runs as the host user rather than as root (see
# `docker-compose.yml`), so the baked cache must be traversable and writable by
# it — huggingface_hub still creates lock files inside the cache directory.
RUN chmod a+x /root && chmod -R a+rwX /root/.cache

# the pipeline is the image's job: one run scrapes, scores and publishes
CMD ["python", "toxic_news/main.py", "update"]
