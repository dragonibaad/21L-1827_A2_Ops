# remote_client.py: runs on localhost, sends images to the cloud model API
import os

import requests
from flask import Flask, request, render_template_string

API_URL = os.environ["REMOTE_API_URL"]              # e.g. http://<PUBLIC_IP>:5000

app = Flask(__name__)

PAGE = """
<h1>VisionGuard (local UI, cloud model)</h1>
<form method="post" action="/check" enctype="multipart/form-data">
  <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
  <button type="submit">Check image</button>
</form>
{% if result %}
  <p><b>{{ result.label }}</b> ({{ "%.1f"|format(result.confidence * 100) }}%)
     by {{ result.model_name }} v{{ result.model_version }}</p>
{% endif %}
{% if error %}<p style="color:#a33">{{ error }}</p>{% endif %}
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/check", methods=["POST"])
def check():
    f = request.files.get("image")
    if f is None:
        return render_template_string(PAGE, error="No file chosen."), 400
    resp = requests.post(
        f"{API_URL}/api/predict",
        files={"image": (f.filename, f.read(), f.mimetype)},
        timeout=60,
    )
    if resp.status_code == 200:
        return render_template_string(PAGE, result=resp.json())
    return render_template_string(PAGE, error=resp.json().get("error", "Remote error")), resp.status_code

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)