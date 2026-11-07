import os

import pandas as pd

from remote_client import call_cloud, init_log

N_PER_CLASS = int(os.environ.get("N_PER_CLASS", "5"))
OUT_CSV = "batch_results.csv"

init_log()
split = pd.read_csv("split_lists.csv")
test = split[split["split"] == "test"]

sample = pd.concat([
    test[test["label"] == 0].sample(n=N_PER_CLASS, random_state=42),  
    test[test["label"] == 1].sample(n=N_PER_CLASS, random_state=42),  
])

results = []
for _, row in sample.iterrows():
    with open(row["file_path"], "rb") as f:
        data = f.read()
    name = os.path.basename(row["file_path"])
    body, error = call_cloud(name, data)

    truth = "AI-Generated" if row["label"] == 1 else "Real"
    predicted = body["label"] if body else "ERROR"
    results.append({
        "file": name,
        "true_label": truth,
        "predicted": predicted,
        "confidence": body["confidence"] if body else None,
        "model_version": body["model_version"] if body else None,
        "correct": predicted == truth,
        "error": error,
    })
    print(f"{name:45s} true={truth:13s} predicted={predicted:13s} correct={predicted == truth}")

out = pd.DataFrame(results)
out.to_csv(OUT_CSV, index=False)
print(f"\n{len(out)} images sent, {out['correct'].mean():.0%} correct. Saved to {OUT_CSV}")