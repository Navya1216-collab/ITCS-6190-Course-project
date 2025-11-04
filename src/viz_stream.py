# src/viz_stream.py
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "outputs/stream_out/avg_by_airline"
OUT_PNG = "outputs/plots/stream_avg_delay.png"

batches = sorted(
    glob.glob(os.path.join(ROOT, "batch=*")),
    key=lambda p: int(p.split("=")[-1])
)
if not batches:
    raise FileNotFoundError(
        f"No snapshots under {ROOT}. Run the streamer first:\n"
        "  python src/04_streaming_demo.py"
    )

latest = batches[-1]
df = pd.read_parquet(latest).sort_values("avg_arr_delay", ascending=False)

plt.figure(figsize=(11, 6))
plt.barh(df["airline"], df["avg_arr_delay"])
plt.xlabel("Average Arrival Delay (minutes)")
plt.ylabel("Airline")
plt.title("Average Arrival Delay by Airline (latest streaming snapshot)")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig(OUT_PNG, dpi=300)
print(f"[OK] saved → {OUT_PNG}")
