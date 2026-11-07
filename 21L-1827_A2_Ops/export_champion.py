# export_champion.py
import json
import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "VisionGuard-classifier"

tracking_uri = os.environ["MLFLOW_TRACKING_URL"]
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")

mlflow.artifacts.download_artifacts(
    f"models:/{MODEL_NAME}@champion",
    dst_path="serving_model",
)

with open("serving_model/model_info.json", "w") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "model_version": int(champion.version),
        "run_id": champion.run_id,
    }, f, indent=2)

print("Exported", MODEL_NAME, "version", champion.version)