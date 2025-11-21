# src/viz_stream_live.py
#
# Live visualization for streaming predictions (Codespaces-friendly)
# Every few seconds it reads all Parquet prediction batches and
# saves a PNG dashboard instead of opening a GUI window.

import os
import time
import glob
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")          # <-- IMPORTANT: non-GUI backend
import matplotlib.pyplot as plt

PRED_DIR = "outputs/stream_out/predictions"
PLOTS_DIR = "outputs/plots"
OUT_PNG = os.path.join(PLOTS_DIR, "live_predictions.png")

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_predictions():
    paths = sorted(glob.glob(f"{PRED_DIR}/batch=*/part-*.parquet"))
    if not paths:
        print("[INFO] No prediction files yet...")
        return None

    dfs = []
    for p in paths:
        try:
            dfs.append(pq.read_table(p).to_pandas())
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    return df


def save_plots(df):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    # --- Subplot 1: Avg actual delay per airline ---
    plt.subplot(1, 2, 1)
    g1 = df.groupby("airline")["arr_delay"].mean().sort_values()
    g1.plot(kind="bar")
    plt.title("Avg Actual Arrival Delay")
    plt.xlabel("Airline")
    plt.ylabel("Minutes")

    # --- Subplot 2: Avg predicted delay probability ---
    plt.subplot(1, 2, 2)
    g2 = df.groupby("airline")["pred_delayed"].mean().sort_values()
    g2.plot(kind="bar")
    plt.title("Avg Predicted Delay Probability")
    plt.xlabel("Airline")
    plt.ylabel("Probability")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"[OK] Saved live plot → {OUT_PNG}")


def main():
    print("\n🚀 Live PNG Visualization Started")
    print(f"Watching prediction files at: {os.path.abspath(PRED_DIR)}")
    print(f"Snapshots written to:        {os.path.abspath(OUT_PNG)}\n")

    while True:
        df = load_predictions()
        if df is not None:
            save_plots(df)
        time.sleep(5)  # refresh every 5 seconds


if __name__ == "__main__":
    main()
