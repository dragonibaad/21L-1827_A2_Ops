FROM python:3.11-slim

WORKDIR /app


COPY Requirements.txt .
RUN pip install -r Requirements.txt

RUN pip install "tf-nightly[and-cuda]==2.23.0.dev20261104"


RUN useradd --create-home --uid 1000 appuser


COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "app.py"]