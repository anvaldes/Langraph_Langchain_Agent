FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# One worker (the dataset and model are loaded once) with threads to serve concurrent requests
CMD exec gunicorn --bind ":${PORT}" --workers 1 --threads 8 --timeout 0 main:app
