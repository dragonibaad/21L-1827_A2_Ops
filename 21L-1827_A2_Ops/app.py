import html
import io
from datetime import datetime, timezone

from flask import Flask, request, redirect, url_for, abort, jsonify, render_template
from PIL import Image
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from inference import load_model_once, predict

MAX_BYTES = 5 * 1024 * 1024
ALLOWED = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

engine = create_engine("sqlite:///visionguard.db")
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file_name = Column(String(120))
    status = Column(String(200))
    label = Column(String(20))
    confidence = Column(Float)

Base.metadata.create_all(engine)
load_model_once()

class UploadError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def validate_upload(file_storage):
    name = secure_filename(file_storage.filename or "")[:120]
    if not name or "." not in name:
        raise UploadError("The file needs a name ending in .jpg, .jpeg or .png.")
    ext = name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        raise UploadError("Unsupported file type. Upload a JPG or PNG.")
    data = file_storage.read(MAX_BYTES + 1)
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadError("The file is larger than 5 MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            actual_format = img.format
    except Exception:
        raise UploadError("This file looks like an image but cannot be decoded.")
    if actual_format != ALLOWED[ext]:
        raise UploadError("The file contents do not match its extension.")
    return name, data

def error_page(message, code):
    return render_template("error.html", message=message), code

@app.route("/")
def upload_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("image")
    db = Session()
    name = secure_filename(file.filename or "")[:120] if file else ""
    try:
        if file is None:
            raise UploadError("No file was sent.")
        name, data = validate_upload(file)
        result = predict(data)
        row = Prediction(file_name=name, status="ok",
                         label=result["label"], confidence=result["confidence"])
        db.add(row)
        db.commit()
        return redirect(url_for("result", pred_id=row.id))
    except UploadError as e:
        db.add(Prediction(file_name=name, status=e.message))
        db.commit()
        return error_page(e.message, 400)
    finally:
        db.close()

@app.route("/result/<int:pred_id>")
def result(pred_id):
    db = Session()
    row = db.get(Prediction, pred_id)
    db.close()
    if row is None or row.status != "ok":
        abort(404)
    return render_template("result.html", row=row)

@app.route("/history")
def history():
    db = Session()
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(50).all()
    db.close()
    return render_template("history.html", rows=rows)

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.errorhandler(413)
def too_large(e):
    return error_page("The file is larger than 5 MB.", 413)

@app.errorhandler(404)
def not_found(e):
    return error_page("That page does not exist.", 404)

@app.errorhandler(500)
def server_error(e):
    return error_page("Something failed on the server. Try again.", 500)

if __name__ == "__main__":
    app.run(debug=False)