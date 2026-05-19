FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY requirements-mvp.txt ./
RUN pip install --no-cache-dir -r requirements-mvp.txt \
    && curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

COPY defapi ./defapi
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "defapi.api:app", "--host", "0.0.0.0", "--port", "8000"]
