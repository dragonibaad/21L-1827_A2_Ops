import os
import sqlite3
import time
from datetime import datetime, timezone

import requests
from flask import Flask, request, render_template_string

API_URL = os.environ["REMOTE_API_URL"].rstrip("/")
LOG_DB = os.environ.get("REMOTE_LOG_DB", "remote_log.db")
TIMEOUT = 60

app = Flask(__name__)


def init_log():
    con = sqlite3.connect(LOG_DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            called_at TEXT NOT NULL,
            file_name TEXT,
            http_status INTEGER,
            latency_ms INTEGER,
            label TEXT,
            confidence REAL,
            model_version INTEGER,
            cloud_id INTEGER,
            error TEXT
        )"""
    )
    con.commit()
    con.close()


def log_call(file_name, http_status, latency_ms, body=None, error=None):
    con = sqlite3.connect(LOG_DB)
    con.execute(
        "INSERT INTO calls (called_at, file_name, http_status, latency_ms, label, "
        "confidence, model_version, cloud_id, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            file_name,
            http_status,
            latency_ms,
            body.get("label") if body else None,
            body.get("confidence") if body else None,
            body.get("model_version") if body else None,
            body.get("id") if body else None,
            error,
        ),
    )
    con.commit()
    con.close()


def elapsed_ms(start):
    return int((time.perf_counter() - start) * 1000)


def call_cloud(file_name, data):
    """Send one image to the cloud API. Returns (body, error_message)."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_URL}/api/predict",
            files={"image": (file_name, data)},
            timeout=TIMEOUT,
        )
        body = resp.json()
    except (requests.RequestException, ValueError) as e:
        log_call(file_name, 0, elapsed_ms(start), error=f"Connection failed: {e}")
        return None, "Could not reach the cloud API."

    latency = elapsed_ms(start)
    if resp.status_code == 200:
        log_call(file_name, 200, latency, body=body)
        return body, None
    message = body.get("error", "Remote error")
    log_call(file_name, resp.status_code, latency, error=message)
    return None, message


STYLE = """
body { font-family: Arial, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 16px; color: #222; }
nav a { margin-right: 16px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin-bottom: 30px; }
th, td { border-bottom: 1px solid #ddd; padding: 6px; text-align: left; }
.err { color: #a33; }
"""
NAV = '<nav><a href="/">Check image</a><a href="/history">History</a></nav><hr>'


def page(body):
    return ("<!doctype html><html><head><meta charset='utf-8'><title>VisionGuard client</title>"
            "<style>" + STYLE + "</style></head><body>" + NAV + body + "</body></html>")


PAGE = page("""
<h2>Check an image</h2>
<form method="post" action="/check" enctype="multipart/form-data">
  <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
  <button type="submit">Submit</button>
</form>
{% if result %}
  <h3>Result</h3>
  <p><b>{{ result.label }}</b>, confidence {{ '%.1f' % (result.confidence * 100) }}%</p>
  <p>Model {{ result.model_name }} v{{ result.model_version }} | cloud record id {{ result.id }}</p>
{% endif %}
{% if error %}<p class="err">{{ error }}</p>{% endif %}
""")

HISTORY = page("""
<h2>Local call log (this laptop)</h2>
<table>
<tr><th>Time (UTC)</th><th>File</th><th>HTTP</th><th>ms</th><th>Label</th><th>Confidence</th><th>Model v</th><th>Cloud id</th><th>Error</th></tr>
{% for r in local %}
<tr><td>{{ r['called_at'] }}</td><td>{{ r['file_name'] }}</td><td>{{ r['http_status'] }}</td><td>{{ r['latency_ms'] }}</td>
<td>{{ r['label'] or '-' }}</td>
<td>{{ '%.1f%%' % (r['confidence'] * 100) if r['confidence'] is not none else '-' }}</td>
<td>{{ r['model_version'] or '-' }}</td><td>{{ r['cloud_id'] or '-' }}</td><td class="err">{{ r['error'] or '' }}</td></tr>
{% else %}
<tr><td colspan="9">No calls yet.</td></tr>
{% endfor %}
</table>

<h2>Cloud database log ({{ api_url }})</h2>
{% if cloud_error %}<p class="err">{{ cloud_error }}</p>{% endif %}
<table>
<tr><th>Time (UTC)</th><th>File</th><th>Status</th><th>Label</th><th>Confidence</th><th>Threshold</th><th>Model v</th></tr>
{% for c in cloud %}
<tr><td>{{ c['created_at'] }}</td><td>{{ c['file_name'] }}</td><td>{{ c['status'] }}</td>
<td>{{ c['label'] or '-' }}</td>
<td>{{ '%.1f%%' % (c['confidence'] * 100) if c['confidence'] is not none else '-' }}</td>
<td>{{ c['threshold'] or '-' }}</td><td>{{ c['model_version'] or '-' }}</td></tr>
{% else %}
<tr><td colspan="7">No cloud records.</td></tr>
{% endfor %}
</table>
""")


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/check", methods=["POST"])
def check():
    f = request.files.get("image")
    if f is None or not f.filename:
        return render_template_string(PAGE, error="Choose an image first.")
    body, error = call_cloud(f.filename, f.read())
    if body:
        return render_template_string(PAGE, result=body)
    return render_template_string(PAGE, error=error)


@app.route("/history")
def history():
    con = sqlite3.connect(LOG_DB)
    con.row_factory = sqlite3.Row
    local = con.execute("SELECT * FROM calls ORDER BY id DESC LIMIT 100").fetchall()
    con.close()
    cloud_error = None
    try:
        cloud = requests.get(f"{API_URL}/api/history", timeout=10).json()
    except (requests.RequestException, ValueError):
        cloud, cloud_error = [], "Could not reach the cloud history endpoint."
    return render_template_string(HISTORY, local=local, cloud=cloud,
                                  cloud_error=cloud_error, api_url=API_URL)


if __name__ == "__main__":
    init_log()
    app.run(host="127.0.0.1", port=5001, debug=False)