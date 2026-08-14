FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

# Railway injects the port to listen on via the $PORT env var at runtime,
# so this must be shell form (not a JSON array) for $PORT to expand.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
