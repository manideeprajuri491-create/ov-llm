import sys
import time
import resource
import psutil
import numpy as np
import openvino_genai as ov_genai

def main():
    model = sys.argv[1]
    threads = int(sys.argv[2])
    hint = sys.argv[3] if len(sys.argv) > 3 else "LATENCY"

    props = {"PERFORMANCE_HINT": hint}

    if threads > 0:
        props["INFERENCE_NUM_THREADS"] = threads

    print(f"Loading model with threads={threads or 'auto'}, hint={hint}")

    pipe = ov_genai.LLMPipeline(model, "CPU", **props)

    prompt = "Explain how a CPU cache works and why it makes programs faster."

    # Warm-up
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

    busy = (
        (sum(proc.cpu_times()[:2]) - cpu0)
        / (time.perf_counter() - wall0)
    )

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    throughput = total_tokens / total_time

    print(
        f"threads={threads or 'auto':>4} "
        f"hint={hint:<10} "
        f"Total={total_time/runs:6.2f}s "
        f"Throughput={throughput:5.2f} tok/s "
        f"busyCores={busy:4.1f} "
        f"peakRAM={peak:5.0f}MB"
        
    )

if __name__ == "__main__":
    main()
