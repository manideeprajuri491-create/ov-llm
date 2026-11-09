import openvino_genai as ov_genai
import time

MODEL_PATH = "models/tinyllama-int4"

prompts = {
    "Short prompt": "What is a CPU cache?",
    "Long prompt": (
        "Explain computer architecture in detail. "
        "Describe the CPU, its arithmetic logic unit, control unit, "
        "registers, cache memory, RAM, ROM, and how these components "
        "work together to execute instructions. "
        "Also explain the importance of memory hierarchy and "
        "how it affects overall computer performance."
    )
}

def main():
    print("Loading TinyLlama INT4...")
    pipe = ov_genai.LLMPipeline(MODEL_PATH, "CPU")

    print("Warming up model...")
    pipe.generate("Hello", max_new_tokens=10)

    print("\nKV Cache Runtime Experiment")
    print("=" * 55)

    for name, prompt in prompts.items():
        start = time.perf_counter()

        answer = pipe.generate(
            prompt,
            max_new_tokens=50
        )

        elapsed = time.perf_counter() - start

        print(f"\n{name}")
        print("-" * 55)
        print(f"Prompt length (characters): {len(prompt)}")
        print(f"Inference time: {elapsed:.2f} seconds")
        print(f"Generated answer:\n{answer}")

    print("\nExperiment completed.")

if __name__ == "__main__":
    main()
