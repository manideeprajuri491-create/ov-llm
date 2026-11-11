
import time
import csv
from datetime import datetime
import openvino_genai as ov_genai

MODEL_PATH = "models/tinyllama-int4"
CSV_FILE = "batch_benchmark_results.csv"

def main():
    pipe = ov_genai.LLMPipeline(MODEL_PATH, "CPU")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 64

    prompts = [
        "Explain what a CPU cache is.",
        "What is the purpose of RAM?",
        "Explain the difference between RAM and ROM.",
        "What is an operating system?",
        "Explain how TCP works.",
        "What is machine learning?",
        "Explain what a compiler does.",
        "What is virtual memory?"
    ]

    print("Warming up...")
    pipe.generate(prompts[:1], cfg)

    print("\nBatch Inference Results")
    print("=" * 55)

    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "model",
            "batch_size",
            "time_seconds",
            "requests_per_second"
        ])

        for n in (1, 2, 4, 8):
            start = time.perf_counter()

            result = pipe.generate(prompts[:n], cfg)

            elapsed = time.perf_counter() - start
            requests_per_sec = n / elapsed if elapsed > 0 else 0

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                MODEL_PATH,
                n,
                round(elapsed, 4),
                round(requests_per_sec, 4)
            ])

            print(
                f"Batch size: {n} | "
                f"Time: {elapsed:.2f}s | "
                f"Requests/sec: {requests_per_sec:.2f}"
            )

    print(f"\nBatch experiment completed. Results saved to {CSV_FILE}")

if __name__ == "__main__":
    main()
