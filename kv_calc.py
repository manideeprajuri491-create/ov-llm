import json
import sys

config_path = f"{sys.argv[1]}/config.json"

with open(config_path) as f:
    cfg = json.load(f)

layers = cfg["num_hidden_layers"]
heads = cfg["num_attention_heads"]
kv_heads = cfg.get("num_key_value_heads", heads)
head_dim = cfg.get("head_dim", cfg["hidden_size"] // heads)
max_ctx = cfg.get("max_position_embeddings")

print("TinyLlama KV Cache Configuration")
print("--------------------------------")
print(f"Layers       : {layers}")
print(f"Attention heads: {heads}")
print(f"KV heads     : {kv_heads}")
print(f"Head dimension: {head_dim}")
print(f"Maximum context: {max_ctx}")

contexts = (256, 512, 1024, 2048, 4096)

print("\nKV Cache Memory Requirements")
print(f"{'dtype':<8}{'B/token':>12}" +
      "".join(f"{c:>12}" for c in contexts))

for name, nbytes in (("f32", 4), ("f16", 2), ("u8", 1)):
    per_tok = 2 * layers * kv_heads * head_dim * nbytes

    print(
        f"{name:<8}{per_tok:>12}" +
        "".join(f"{per_tok * c / 2**20:>12.1f}" for c in contexts)
    )
