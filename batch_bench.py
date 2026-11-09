import time
import openvino_genai as ov_genai

MODEL_PATH = "models/tinyllama-int4"

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

    for n in (1, 2, 4, 8):
        start = time.perf_counter()

        result = pipe.generate(prompts[:n], cfg)

        elapsed = time.perf_counter() - start

        print(
            f"Batch size: {n} | "
            f"Time: {elapsed:.2f}s | "
            f"Requests/sec: {n / elapsed:.2f}"
        )

    print("\nBatch experiment completed.")

if __name__ == "__main__":
    main()
