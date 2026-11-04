import os
import tensorflow as tf
import mlflow.keras

IMG_SIZE = 224
CLASS_NAMES = {0: "Real", 1: "AI-Generated"}
MODEL_DIR = os.environ.get("VISIONGUARD_MODEL_DIR", "serving_model")

_model = None

def load_model_once():
    global _model
    if _model is None:
        _model = mlflow.keras.load_model(MODEL_DIR)
    return _model

def preprocess(image_bytes):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)

def predict(image_bytes):
    model = load_model_once()
    prob_ai = float(model.predict(preprocess(image_bytes), verbose=0)[0][0])
    label_id = 1 if prob_ai >= 0.5 else 0
    confidence = prob_ai if label_id == 1 else 1.0 - prob_ai
    return {
        "label": CLASS_NAMES[label_id],
        "confidence": round(confidence, 4),
        "prob_ai": round(prob_ai, 4),
    }