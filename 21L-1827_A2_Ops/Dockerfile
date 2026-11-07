FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*




COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir tensorflow-cpu==2.17.1



COPY app.py inference.py config.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY serving_model/ ./serving_model/



RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV VISIONGUARD_MODEL_DIR=/app/serving_model \
    VISIONGUARD_HOST=0.0.0.0 \
    VISIONGUARD_PORT=5000 \
    VISIONGUARD_DEBUG=0

EXPOSE 5000


HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)"

CMD ["python", "app.py"]