FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

# TensorFlow installed separately
RUN pip install --no-cache-dir tensorflow-cpu==2.17.1

# Verify dependency compatibility
RUN pip check

# Copy application files
COPY app.py inference.py config.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY serving_model/ ./serving_model/

# Create non-root user and database directory
RUN useradd --create-home --uid 1001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

# Run application as non-root user
USER appuser

# Application configuration
ENV VISIONGUARD_MODEL_DIR=/app/serving_model \
    VISIONGUARD_HOST=0.0.0.0 \
    VISIONGUARD_PORT=5000 \
    VISIONGUARD_DEBUG=0

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)"

# Start application
CMD ["python", "app.py"]