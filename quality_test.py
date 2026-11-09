
import openvino_genai as ov_genai
import json

def main():
    models = {
        "INT4": "models/tinyllama-int4",
        "INT8": "models/tinyllama-int8",
        "FP16": "models/tinyllama-fp16"
    }

    questions = [
        "What is a CPU cache? Explain its purpose.",
        "Explain the difference between RAM and ROM.",
        "What is machine learning? Give two examples.",
        "Explain how TCP works in computer networks.",
        "What is the difference between a compiler and an interpreter?"
    ]

    results = {}

    for name, model_path in models.items():
        print("\n" + "=" * 55)
        print(f"Testing {name}")
        print("=" * 55)

        print("Loading model...")
        pipe = ov_genai.LLMPipeline(model_path, "CPU")

        results[name] = []

        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")

            answer = pipe.generate(
                question,
                max_new_tokens=100
            )

            print(f"\nAnswer: {answer}")

            results[name].append({
                "question": question,
                "answer": answer
            })

        del pipe

    with open("quality_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nQuality testing completed.")
    print("Results saved to quality_results.json")

if __name__ == "__main__":
    main()
