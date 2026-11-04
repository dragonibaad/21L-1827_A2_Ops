import pandas as pd
from inference import load_model_once, predict

split = pd.read_csv("split_lists.csv")
test = split[split["split"] == "test"]
real_path = test[test["label"] == 0]["file_path"].iloc[0]
ai_path = test[test["label"] == 1]["file_path"].iloc[0]

load_model_once()
for path, truth in [(real_path, "Real"), (ai_path, "AI-Generated")]:
    with open(path, "rb") as f:
        result = predict(f.read())
    print(f"{path}")
    print(f"  true: {truth} | predicted: {result['label']} | confidence: {result['confidence']}")