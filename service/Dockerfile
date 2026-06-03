FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; add chromium libs only if you enable render_js.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# mlruns is mounted/volumed in so the container sees the registered model.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
