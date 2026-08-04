FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT; Hugging Face Spaces (Docker SDK) expects port 7860.
# This default works for both — Render overrides $PORT automatically,
# and we default to 7860 for Spaces.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
