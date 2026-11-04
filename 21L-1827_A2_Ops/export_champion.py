import dagshub                                           
import mlflow

dagshub.init(repo_owner="dragonibaad", repo_name="VisionGuard", mlflow=True)
mlflow.artifacts.download_artifacts(
    "models:/VisionGuard-classifier@champion",
    dst_path="21L-1827_A2_Ops/serving_model",
)
print("Exported champion to ./serving_model")