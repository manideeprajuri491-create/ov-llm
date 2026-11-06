Core Concepts of Local LLM Inference

## 1. Tokens and Tokenizer

A token is a piece of text represented by an integer ID that a language model can process. A tokenizer converts text into token IDs, and a detokenizer converts generated token IDs back into readable text. Different words, symbols, and languages can produce different numbers of tokens.


## 2. Prefill vs. Decode

Prefill processes the input prompt and computes the initial attention information needed for generation. Decode generates the answer one token at a time, reusing the KV cache from earlier tokens. Time to First Token (TTFT) measures the wait until the first output token, while TPOT measures the average time per output token after generation begins.

## 3. INT4 Weight Compression

FP16 stores each weight using 2 bytes, whereas INT4 uses 4 bits, or half a byte, per quantized weight before additional overhead. For 1.1 billion parameters, the approximate weight sizes are 2.2 GB for FP16 and 0.55 GB for INT4. INT4 can reduce memory requirements and data movement, but may affect output quality and does not always guarantee faster inference.


## 4. What Happens Between Typing a Prompt and Seeing the First Word?

When a user types a prompt, the tokenizer converts the text into token IDs. During prefill, the model processes the prompt and builds the initial KV cache. The model then begins decoding and selects the first output token. The detokenizer converts that token into readable text, which the application displays to the user. Additional tokens are generated iteratively until the response finishes.

Benchmark Results

Requested Tokens

Generated Tokens

TTFT (ms)

TPOT (ms/token)

Throughput (tokens/sec)

50

50

293.5, 304.9

71.0, 71.6

14.1, 14.0

100

100

290.9, 281.6

67.0, 67.4

14.9, 14.8

300

126

231.7, 253.0

65.6, 66.0

15.2, 15.2

Observations

TTFT remained relatively stable because the same input prompt was used for all experiments.

TPOT remained relatively consistent, ranging from 65.6 to 71.6 milliseconds per token.

The model achieved approximately 14–15.2 tokens per second on CPU.

In the 300-token experiment, the model generated only 126 tokens because generation stopped before reaching the maximum token limit.

Conclusion

The TinyLlama INT4 model successfully performed CPU inference using OpenVINO GenAI. The benchmark demonstrated stable generation performance with approximately 14–15 tokens per second. These results will serve as the baseline for future INT8 and FP16 quantization comparisons.

--- Performance Metrics ---
TTFT: 3511.94 ms
TPOT: 46.3413 ms/token
Throughput: 21.579 tokens/sec