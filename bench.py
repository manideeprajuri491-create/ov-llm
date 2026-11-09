
import openvino_genai as ov_genai
import time
import numpy as np

def benchmark():
    models = {
        "INT4": "models/tinyllama-int4",
        "INT8": "models/tinyllama-int8",
        "FP16": "models/tinyllama-fp16"
    }

    prompt = "What is a CPU cache?"
    token_counts = (50, 100, 300)

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

            total_time = time.perf_counter() - start

            tokenized = tokenizer.encode(result)
            generated_tokens = int(np.prod(tokenized.input_ids.shape))

            throughput = generated_tokens / total_time if total_time > 0 else 0

            print(
                f"Requested: {n:3} | "
                f"Generated: {generated_tokens:3} | "
                f"Total: {total_time:6.2f} sec | "
                f"Throughput: {throughput:5.2f} tokens/sec"
            )

        del pipe

    print("\nBenchmark completed.")

if __name__ == "__main__":
    benchmark()
