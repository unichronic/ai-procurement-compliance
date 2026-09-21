# Container for the standards recommendation and tender-audit engine.
#
# The repo's root Dockerfile builds the team's separate FastAPI app and does
# not include backend/ or data_pipeline/ at all, so this one exists rather
# than overloading it.
#
# Two-stage: the embedding model and the document embeddings are baked in at
# build time. Encoding 6,383 documents takes ~50s, and a container that pays
# that on every start fails its own readiness probe during a rolling deploy.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


FROM python:3.11-slim

# tesseract is a runtime dependency of the OCR fallback for scanned tenders.
# Without it /lint/upload still works, it just can't read scans.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    USE_EMBEDDING_CACHE=1 \
    EMBEDDING_CACHE_DIR=/opt/embeddings

WORKDIR /app
COPY backend/ /app/
COPY data_pipeline/ /app/data_pipeline/

# Bake the sentence-transformer weights in, so the container neither reaches
# the network on first request nor ships without a model.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('embedding model cached')"

# Warm the document embeddings so the first request is fast and readiness
# passes immediately.
RUN python -c "\
import json; \
from app.core.retrieval import StandardsIndex; \
s = json.load(open('app/data/standards.json'))['standards']; \
i = StandardsIndex(s); i.build(); \
print(f'warmed embeddings for {len(s)} standards')"

# Non-root: nothing here needs to write outside the baked cache.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /opt/embeddings /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
