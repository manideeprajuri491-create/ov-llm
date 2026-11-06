
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("models/tinyllama-int4", "CPU")

print("Warming up model...")
pipe.generate(["Hello"], max_new_tokens=10)

prompt = "What is a CPU cache?"

print("\nTinyLlama INT4 CPU Benchmark\n")

for n in (50, 100, 300):
    res = pipe.generate([prompt], max_new_tokens=n)
    m = res.perf_metrics

    print(
        f"Requested: {n:3} | "
        f"Generated: {m.get_num_generated_tokens():3} | "
        f"TTFT: {m.get_ttft().mean:7.1f} ms | "
        f"TPOT: {m.get_tpot().mean:6.1f} ms/token | "
        f"Throughput: {m.get_throughput().mean:5.1f} tokens/s"
    )
