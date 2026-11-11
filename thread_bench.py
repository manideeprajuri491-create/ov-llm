
import sys
import time
import resource
import psutil
import numpy as np
import csv
from datetime import datetime
import openvino_genai as ov_genai

def main():
    if len(sys.argv) < 3:
        print("Usage: python thread_bench.py MODEL_PATH THREADS [PERF_HINT]")
        sys.exit(1)

    model = sys.argv[1]
    threads = int(sys.argv[2])
    hint = sys.argv[3] if len(sys.argv) > 3 else "LATENCY"

    props = {"PERFORMANCE_HINT": hint}

    if threads > 0:
        props["INFERENCE_NUM_THREADS"] = threads

    print(f"Loading model with threads={threads or 'auto'}, hint={hint}")

    pipe = ov_genai.LLMPipeline(model, "CPU", **props)

    prompt = "Explain how a CPU cache works and why it makes programs faster."

    print("Warming up...")
    pipe.generate(prompt, max_new_tokens=128)

    tokenizer = pipe.get_tokenizer()
    proc = psutil.Process()

    runs = 4
    total_time = 0.0
    total_tokens = 0

    cpu0 = sum(proc.cpu_times()[:2])
    wall0 = time.perf_counter()

    for _ in range(runs):
        start = time.perf_counter()

        answer = pipe.generate(prompt, max_new_tokens=128)

        elapsed = time.perf_counter() - start
        tokens = int(np.prod(tokenizer.encode(answer).input_ids.shape))

        total_time += elapsed
        total_tokens += tokens

    wall_elapsed = time.perf_counter() - wall0

    busy = (
        (sum(proc.cpu_times()[:2]) - cpu0)
        / wall_elapsed
    ) if wall_elapsed > 0 else 0

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_time = total_time / runs

    print(
        f"threads={threads or 'auto':>4} "
        f"hint={hint:<10} "
        f"Total={avg_time:6.2f}s "
        f"Throughput={throughput:5.2f} tok/s "
        f"busyCores={busy:4.1f} "
        f"peakRAM={peak:5.0f}MB"
    )

    csv_file = "thread_benchmark_results.csv"

    write_header = True
    try:
        with open(csv_file, "r"):
            write_header = False
    except FileNotFoundError:
        pass

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "timestamp",
                "model",
                "threads",
                "performance_hint",
                "average_time_seconds",
                "throughput_tokens_per_sec",
                "cpu_busy_cores",
                "peak_ram_mb"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            model,
            threads or "auto",
            hint,
            round(avg_time, 4),
            round(throughput, 4),
            round(busy, 2),
            round(peak, 2)
        ])

    print(f"\nResults appended to {csv_file}")

if __name__ == "__main__":
    main()
