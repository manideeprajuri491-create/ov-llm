models = {
    "TinyLlama-1.1B": {
        "layers": 22,
        "kv_heads": 4,
        "head_dim": 64
    },
    "Qwen2.5-1.5B": {
        "layers": 28,
        "kv_heads": 2,
        "head_dim": 128
    },
    "Llama-3.2-3B": {
        "layers": 28,
        "kv_heads": 8,
        "head_dim": 128
    }
}

contexts = [512, 1024, 2048]
precisions = {
    "FP32": 4,
    "FP16": 2,
    "UINT8": 1
}

for name, cfg in models.items():
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    for dtype, bytes_per_element in precisions.items():
        per_token = (
            2 * cfg["layers"] *
            cfg["kv_heads"] *
            cfg["head_dim"] *
            bytes_per_element
        )

        print(f"\n{dtype}:")
        print(f"Bytes per token: {per_token}")

        for ctx in contexts:
            memory = per_token * ctx / (1024 ** 2)
            print(f"{ctx} tokens: {memory:.2f} MiB")
