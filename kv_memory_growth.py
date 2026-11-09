layers = 22
kv_heads = 4
head_dim = 64

precisions = {
    "FP32": 4,
    "FP16": 2,
    "INT8": 1
}

contexts = [128, 256, 512, 1024, 2048]

print("TinyLlama KV Cache Memory Growth")
print("=" * 65)

for dtype, bytes_per_element in precisions.items():

    print(f"\nPrecision: {dtype}")
    print("-" * 65)

    per_token = (
        2 * layers * kv_heads *
        head_dim * bytes_per_element
    )

    print(f"Memory per token: {per_token} bytes")

    for context in contexts:

        memory_mb = (
            per_token * context / (1024 ** 2)
        )

        print(
            f"Context: {context:4} tokens | "
            f"Memory: {memory_mb:7.2f} MiB"
        )
