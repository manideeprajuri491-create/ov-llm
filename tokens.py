from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/tinyllama-int4")
text = "What is a CPU cache?"
ids = tok(text)["input_ids"]

print("Text:", text)
print("Token IDs:", ids)
print("Tokens:", tok.convert_ids_to_tokens(ids))
print("Characters:", len(text))
print("Token count:", len(ids))