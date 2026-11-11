
import openvino_genai as ov_genai
import time
import numpy as np
import csv
from datetime import datetime

def benchmark():
    models = {
        "INT4": "models/tinyllama-int4",
        "INT8": "models/tinyllama-int8",
        "FP16": "models/tinyllama-fp16"
    }

    prompt = "What is a CPU cache?"
    token_counts = (50, 100, 300)

    csv_file = "benchmark_results.csv"

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "model",
            "requested_tokens",
            "estimated_generated_tokens",
            "time_seconds",
            "throughput_tokens_per_sec"
        ])

        for name, model_path in models.items():
            print(f"\n{'='*50}")
            print(f"TinyLlama {name} CPU Benchmark")
            print(f"{'='*50}")

            print("Loading model...")
            pipe = ov_genai.LLMPipeline(model_path, "CPU")

            print("Warming up model...")
            pipe.generate("Hello", max_new_tokens=10)

            tokenizer = pipe.get_tokenizer()

            for n in token_counts:
                start = time.perf_counter()

                result = pipe.generate(
                    prompt,
                    max_new_tokens=n
                )

                elapsed = time.perf_counter() - start

                tokenized = tokenizer.encode(result)
                generated_tokens = int(
                    np.prod(tokenized.input_ids.shape)
                )

                throughput = (
                    generated_tokens / elapsed
                    if elapsed > 0 else 0
                )

                timestamp = datetime.now().isoformat(
                    timespec="seconds"
                )

                writer.writerow([
                    timestamp,
                    name,
                    n,
                    generated_tokens,
                    round(elapsed, 4),
                    round(throughput, 4)
                ])

                print(
                    f"Requested: {n:3} | "
                    f"Generated: {generated_tokens:3} | "
                    f"Time: {elapsed:6.2f}s | "
                    f"Throughput: {throughput:5.2f} tok/s"
                )

            del pipe

    print(f"\nBenchmark completed. Results saved to {csv_file}")


if __name__ == "__main__":
    benchmark()
