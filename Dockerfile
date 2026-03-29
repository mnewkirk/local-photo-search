# Placeholder — will be fleshed out in M7
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Photo library mounted read-only, database read-write
VOLUME ["/photos", "/data"]

ENTRYPOINT ["python", "cli.py"]
