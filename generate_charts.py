
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)

# 1. Model comparison
model_df = pd.read_csv("benchmark_results.csv")

plt.figure(figsize=(9, 5))

for model in model_df["model"].unique():
    data = model_df[model_df["model"] == model]
    plt.plot(
        data["requested_tokens"],
        data["throughput_tokens_per_sec"],
        marker="o",
        label=model
    )

plt.title("TinyLlama Model Throughput Comparison")
plt.xlabel("Requested Maximum Tokens")
plt.ylabel("Throughput (tokens/sec)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=300)
plt.close()


# 2. Batch throughput
batch_df = pd.read_csv("batch_benchmark_results.csv")

plt.figure(figsize=(8, 5))
plt.plot(
    batch_df["batch_size"],
    batch_df["requests_per_second"],
    marker="o"
)

plt.title("Batch Inference Throughput")
plt.xlabel("Batch Size")
plt.ylabel("Requests per Second")
plt.xticks(batch_df["batch_size"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("charts/batch_throughput.png", dpi=300)
plt.close()


# 3. Thread throughput (only if multiple configurations exist)
thread_file = "thread_benchmark_results.csv"

if os.path.exists(thread_file):
    thread_df = pd.read_csv(thread_file)

    if thread_df["threads"].nunique() >= 2:
        thread_df = thread_df.copy()
        thread_df["threads"] = thread_df["threads"].astype(str)

        plt.figure(figsize=(8, 5))
        plt.bar(
            thread_df["threads"],
            thread_df["throughput_tokens_per_sec"]
        )

        plt.title("CPU Thread Performance Comparison")
        plt.xlabel("Thread Configuration")
        plt.ylabel("Throughput (tokens/sec)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("charts/thread_throughput.png", dpi=300)
        plt.close()

        print("Created thread_throughput.png")
    else:
        print("Skipping thread chart: need at least two thread configurations.")


print("Charts generated successfully!")
print("Saved inside the charts/ directory.")
