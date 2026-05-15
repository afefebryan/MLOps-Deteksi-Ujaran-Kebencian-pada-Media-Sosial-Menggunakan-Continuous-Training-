FROM python:3.12-slim

WORKDIR /app

# install dependencies sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# copy dan install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

# copy source code
COPY src/train/app.py .
COPY src/train/train.py .

# port API
EXPOSE 8000

# jalankan FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]