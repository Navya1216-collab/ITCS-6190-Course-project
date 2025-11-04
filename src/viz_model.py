# src/viz_model.py
import os
import pandas as pd
import matplotlib.pyplot as plt

MODEL_DIR = "outputs/models"
CSV_PATH  = os.path.join(MODEL_DIR, "feature_importances_rf.csv")
OUT_PNG   = "outputs/plots/feature_importance_rf.png"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"{CSV_PATH} not found. Train RF first:\n"
        "  python src/03_predictive_model.py --task classify --algo rf --tree_max_bins 4096 --curated_dir ./outputs/curated"
    )

df = pd.read_csv(CSV_PATH).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df["feature"], df["importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance — Random Forest (Delay > 15 min)")
plt.tight_layout()
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig(OUT_PNG, dpi=300)
print(f"[OK] saved → {OUT_PNG}")
