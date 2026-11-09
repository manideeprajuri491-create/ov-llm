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

# Week 3: TinyLlama Model Quantization and Performance Comparison

## 1. Objective

The objective was to study different model precision formats and compare their storage requirements, CPU inference speed, and generated answer quality using OpenVINO GenAI.

## 2. Models Used

The same base model, TinyLlama 1.1B Chat, was exported into three formats:

- FP16 – 16-bit floating-point precision.
- INT8 – 8-bit integer weight quantization.
- INT4 – 4-bit integer weight quantization.

## 3. Model Size Comparison

| Model | Storage |
|---|---:|
| INT4 | 632 MB |
| INT8 | 1.1 GB |
| FP16 | 2.1 GB |

INT4 required the least storage, while FP16 required the most.

## 4. CPU Performance Comparison

Average measured throughput:

| Model | Throughput |
|---|---:|
| INT4 | 29.04 tokens/sec |
| INT8 | 20.41 tokens/sec |
| FP16 | 11.68 tokens/sec |

INT4 achieved the highest measured throughput, approximately 2.49 times that of FP16.

## 5. Quality Evaluation

Five common technical questions were used to evaluate the models.

The evaluation considered:
- Factual correctness
- Relevance
- Completeness
- Coherence
- Consistency

All three models generated relevant answers. INT8 and FP16 showed similar answer quality in several questions. FP16 provided a more accurate explanation of TCP connection establishment in the tested responses. Some answers from all models were incomplete due to the generation token limit.

## 6. Conclusion

The experiment demonstrated the trade-offs between model precision, storage requirements, inference speed, and output quality.

INT4 provided the smallest model size and highest measured CPU throughput. INT8 offered an intermediate balance, while FP16 required the most storage and had the lowest measured throughput.

For resource-constrained CPU environments, INT4 can be a practical choice when inference speed and storage efficiency are important. However, the final choice should also consider answer quality and the application's requirements.

Further repeated benchmarks and standardized quality evaluations would improve the reliability of these findings.
## Week 4 - Day 1: Understanding KV Cache

### Why KV Cache Exists
The KV cache stores previously calculated Key and Value vectors
during LLM inference. Without caching, the model would repeatedly
calculate these vectors for previous tokens. With caching, it
calculates them only for new tokens and reuses previous values.

The advantage is reduced repeated computation and faster
autoregressive generation. The disadvantage is additional memory
consumption.

### KV Cache Formula
KV bytes per token =
2 * layers * KV heads * head dimension * bytes per element

KV cache size =
KV bytes per token * number of context tokens

### TinyLlama Calculation
Layers = 22
Attention heads = 32
KV heads = 4
Head dimension = 64
FP16 = 2 bytes

KV bytes per token =
2 * 22 * 4 * 64 * 2 = 22528 bytes

At 2048 tokens:
22528 * 2048 = 44 MiB approximately.

### Grouped-Query Attention (GQA)
TinyLlama uses 32 attention heads and 4 KV heads.
Eight attention heads share each KV head.
GQA reduces KV cache memory requirements.

Without GQA, the cache would be approximately
352 MiB at 2048 tokens instead of 44 MiB.

### Prefill and Decode
Prefill processes the input prompt and creates the initial
KV cache. It affects Time To First Token (TTFT).

Decode generates tokens one by one and reuses the KV cache.
It affects Time Per Output Token (TPOT). As context grows,
each decode step has more cached information to read.

### Day 2: KV Cache Size Calculations

KV cache memory increases linearly with context length.

FP32 uses 4 bytes per element.
FP16 uses 2 bytes per element.
UINT8 uses 1 byte per element.

The KV cache memory is directly proportional to:
- Number of transformer layers
- Number of KV heads
- Head dimension
- Number of context tokens
- Bytes per element

The INT4 TinyLlama model weights occupy approximately 632 MB.
Its FP16 KV cache at 2048 tokens is approximately 44 MiB.

For TinyLlama, model weights occupy considerably more memory
than the KV cache in a single-user, 2048-token scenario.

However, KV cache memory becomes more important with larger
models, longer contexts, and multiple simultaneous users.

### Day 2: Comparison of Different LLMs

Compared theoretical KV cache requirements for:

1. TinyLlama 1.1B
2. Qwen2.5 1.5B
3. Llama 3.2 3B

FP16 KV cache at 2048 tokens:

TinyLlama: 44 MiB
Qwen2.5: 56 MiB
Llama 3.2: 224 MiB

Larger models can require more KV cache memory.
The exact requirement depends on the number of layers,
KV heads, head dimension, cache precision and context length.

These are theoretical estimates, not measured runtime memory.

### Day 3: OpenVINO KV Cache Inspection

Inspected the exported TinyLlama OpenVINO model using
the OpenVINO Python API.

Model inputs:
- input_ids
- attention_mask
- position_ids
- beam_idx

Model output:
- logits

The main model graph did not expose explicitly named
past_key_values inputs or outputs.

This does not prove that KV caching is absent.
OpenVINO GenAI can manage cache state internally
during autoregressive generation.

The beam_idx input is associated with beam-search
cache reordering and is not itself the KV cache.

The LLMPipeline manages the generation process.

### Day 3: Stateful OpenVINO Model

Inspected the internal OpenVINO graph.

Found:
- 44 ReadValue operations
- 44 Assign operations
- 4229 total graph operations

ReadValue retrieves previously stored state.
Assign updates the stored state.

TinyLlama has 22 transformer layers.
Each layer maintains one Key cache and one Value cache.

Therefore:
22 * 2 = 44 KV cache state variables.

The graph inspection confirms that the exported model
contains stateful operations consistent with KV caching.

OpenVINO GenAI manages cache handling during generation.
### KV Cache State Variable Inspection

Inspected the OpenVINO model using model.get_variables().

Results:
- Total state variables: 44
- Key cache variables: 22
- Value cache variables: 22
- Cache data type: FP32
- Cache shape: [?, 4, ?, 64]

Each transformer layer contains one Key cache
and one Value cache variable.

The dynamic dimensions represent batch size
and sequence length.

Important observation:
Although the model weights are quantized to INT4,
the inspected KV cache variables use FP32.
Weight precision and KV cache precision are independent.

### KV Cache Runtime Experiment

Model: TinyLlama INT4
Device: CPU
Requested output: 50 tokens

Experiment results:

Short prompt:
- Prompt length: 20 characters
- Inference time: 1.56 seconds

Long prompt:
- Prompt length: 295 characters
- Inference time: 1.82 seconds

Observation:
The longer prompt took 0.26 seconds more
than the shorter prompt in this run.

Reason:
A longer prompt generally requires more
computation during the prefill stage and
creates a larger KV cache.

During decoding, previously calculated
Key and Value vectors are reused instead
of being recomputed.

Limitation:
This experiment measures total inference
time, not prefill and decode time separately.

### KV Cache Memory Growth Experiment

Model: TinyLlama 1.1B
Layers: 22
KV heads: 4
Head dimension: 64
Batch size: 1

Memory formula:

M = 2 * L * Hkv * D * B * S

Results at 2048 tokens:

FP32: 88 MiB
FP16: 44 MiB
INT8: 22 MiB

Observations:

1. KV cache memory grows linearly with context length.
2. FP16 requires 50% less memory than FP32.
3. INT8 requires 75% less memory than FP32.
4. INT4 weight quantization does not automatically
   quantize the KV cache.

The inspected TinyLlama INT4 OpenVINO model
uses FP32 KV cache variables.

These calculations represent theoretical
KV cache memory, not total application memory.
# WEEK 4: KV CACHE INVESTIGATION

## 1. Objective

To understand how KV cache works in transformer-based
LLMs and investigate its memory requirements,
internal representation and optimization techniques.

## 2. Model Configuration

Model: TinyLlama-1.1B-Chat-v1.0

Transformer layers: 22
Attention heads: 32
KV heads: 4
Head dimension: 64
Maximum context length: 2048

## 3. KV Cache Formula

KV Cache Memory = 2 * L * Hkv * D * B * S

Where:
L = Number of transformer layers
Hkv = Number of KV heads
D = Head dimension
B = Bytes per element
S = Number of cached tokens

For TinyLlama FP32:

Memory per token = 45056 bytes

## 4. OpenVINO Graph Inspection

Inspected the OpenVINO model using:

- model.inputs
- model.outputs
- model.get_ordered_ops()
- model.get_variables()

Results:

ReadValue operations: 44
Assign operations: 44
State variables: 44

The state variable IDs identify:

22 Key cache variables
22 Value cache variables

Cache variable shape: [?, 4, ?, 64]
Cache data type: FP32

The dynamic dimensions represent batch size
and sequence length.

## 5. Runtime Experiment

Compared two prompts using TinyLlama INT4.

Short prompt:
20 characters
Inference time: 1.56 seconds

Long prompt:
295 characters
Inference time: 1.82 seconds

Observation:

The longer prompt took more time in this run.
Longer prompts generally require more prefill
computation.

This experiment measured total inference time,
not prefill and decode time separately.

## 6. KV Cache Memory Comparison

At 2048 tokens, batch size 1:

FP32: 88 MiB
FP16: 44 MiB
INT8: 22 MiB

Observations:

- Memory grows linearly with context length.
- FP16 uses 50% less memory than FP32.
- INT8 uses 75% less memory than FP32.
- Weight quantization does not automatically
  quantize the KV cache.

These are theoretical memory calculations.

## 7. KV Cache Optimization Techniques

FP16 KV Cache:
Uses 16-bit floating-point values to reduce
memory requirements.

INT8 KV Cache:
Uses 8-bit values to further reduce memory,
with quantization and dequantization.

PagedAttention:
Divides KV cache allocation into blocks to
improve memory utilization.

## 8. Final Findings

1. KV cache avoids recomputing previous Key and
   Value vectors during autoregressive generation.

2. TinyLlama has 22 transformer layers and
   44 KV cache state variables.

3. The inspected INT4 OpenVINO model uses
   FP32 KV cache variables.

4. KV cache memory increases with context length.

5. Weight precision and KV cache precision
   are independent.

6. Cache precision and memory allocation
   techniques can improve inference efficiency.

## 9. Limitations

- Runtime experiment measured total inference time.
- FP16 and INT8 cache memory values were theoretical.
- KV cache quantization was not implemented.
- PagedAttention was studied conceptually,
  not implemented in this project.

## 10. Conclusion

The investigation demonstrated how OpenVINO
represents KV cache through state variables and
how cache memory depends on model architecture,
precision and context length.

Understanding KV cache is important for optimizing
memory consumption and improving LLM inference
performance.


## Week 5 - Hardware and Performance

### Day 1: Hardware Information

Processor: 12th Gen Intel Core i5-12450H
Physical cores: 6
Logical CPUs: 12
Threads per core: 2
L2 cache: 7.5 MiB
L3 cache: 12 MiB

WSL memory:
- Total RAM: 7.6 GiB
- Available RAM during inspection: 7.1 GiB
- Swap: 2.0 GiB
- Available logical CPUs: 12

OpenVINO detected devices:
- CPU: 12th Gen Intel Core i5-12450H
- GPU/NPU: Not detected

### Compute-bound vs Memory-bound

- Prefill processes the input prompt and performs substantial computation. It is often compute-bound.
- Decode generates tokens one by one and repeatedly reads model weights and KV-cache data. It is often memory-bandwidth-bound.
- Increasing CPU threads can improve performance until memory bandwidth becomes a bottleneck.

### Why Hardware Matters

CPU cores, cache sizes, RAM capacity, and memory bandwidth influence LLM inference performance. We will investigate these factors through experiments in the following days.

## Week 5 - Day 2: Memory Bandwidth

Measured hardware details:
- Approximate single-threaded NumPy read bandwidth: 16.8 GB/s.
- Array size: 256 MB.
- Measurement: best result across 10 reads.

Theoretical decode speed estimates:
- INT4 (0.632 GB): 26.58 tokens/sec.
- INT8 (1.1 GB): 15.27 tokens/sec.
- FP16 (2.1 GB): 8.00 tokens/sec.

These are rough bandwidth-based estimates, not strict upper
bounds. Actual inference also involves computation, KV-cache
access, dequantization, CPU caching and other overheads.

The Week 3 benchmark measured average throughput of:
- INT4: 29.04 tokens/sec.
- INT8: 20.41 tokens/sec.
- FP16: 11.68 tokens/sec.

The single-thread bandwidth estimate is not directly comparable
to multithreaded inference throughput.


## Week 5 - Day 3: CPU Thread Scaling

Tested TinyLlama INT4 on Intel Core i5-12450H using OpenVINO CPU.

| Threads | Generation time | Throughput | Busy cores | Peak RAM |
|---|---:|---:|---:|---:|
| 1 | 8.23 s | 15.68 tok/s | 1.0 | 1367 MB |
| 2 | 5.37 s | 24.03 tok/s | 2.0 | 1368 MB |
| 4 | 3.99 s | 32.30 tok/s | 4.0 | 1367 MB |
| 8 | 4.19 s | 30.80 tok/s | 5.9 | 1368 MB |
| Auto | 4.16 s | 30.98 tok/s | 5.9 | 1368 MB |

Observations:
- Four threads produced the highest measured throughput.
- Throughput improved significantly from 1 to 4 threads.
- Eight threads and automatic allocation did not improve performance.
- Peak RAM remained approximately 1.37 GB across configurations.
- Additional threads can compete for shared CPU cache and memory bandwidth.

Four threads are the best configuration observed in this experiment.
Results are preliminary single-run measurements.


### Performance Hint Comparison

Tested TinyLlama INT4 with automatic thread allocation.

| Metric | LATENCY | THROUGHPUT |
|---|---:|---:|
| Generation time | 4.43 s | 4.70 s |
| Throughput | 29.10 tok/s | 27.47 tok/s |
| Busy CPU cores | 6.0 | 3.0 |
| Peak RAM | 1368 MB | 1628 MB |

Observations:
- LATENCY achieved higher throughput in this single-request test.
- THROUGHPUT used more peak RAM in this experiment.
- A single-request test does not fully demonstrate the benefit of
  the THROUGHPUT hint.
- Results are preliminary and may vary between runs.


### Batch Inference Experiment

Tested batch inference with TinyLlama INT4.

| Batch size | Total time | Requests/sec |
|---|---:|---:|
| 1 | 2.21 s | 0.45 |
| 2 | 2.63 s | 0.76 |
| 4 | 3.60 s | 1.11 |
| 8 | 5.58 s | 1.43 |

Observations:
- Request throughput increased with batch size.
- Batch size 8 achieved 1.43 requests/sec.
- Batch size 8 achieved approximately 3.18 times the
  request throughput of batch size 1.
- Total processing time increased with batch size.
- Requests/sec and generated tokens/sec are different metrics.
- Individual request latency and memory usage were not
  separately measured in this experiment.


### Day 5: Hardware Utilization Monitoring

Monitored TinyLlama INT4 inference using htop.

Benchmark configuration:
- CPU threads: 4
- Performance hint: LATENCY

Measured results:
- Average generation time: 4.07 seconds
- Throughput: 31.66 tokens/sec
- Busy CPU cores: 4.0
- Peak process RAM: 1368 MB
- htop RAM observed during inference: approximately 1.16 GB
- Swap usage: 0 GB

Observations:
- Multiple logical CPUs reached approximately 99-100% utilization.
- Inference actively uses multiple CPU cores.
- Memory consumption remained within available WSL RAM.
- OpenVINO detected only the CPU device.
- GPU and NPU experiments were unavailable in this WSL environment.


# Week 5: Hardware Requirements and Performance Analysis

## 1. Hardware Environment

The experiments were conducted in Ubuntu WSL.

- Processor: 12th Gen Intel Core i5-12450H
- Physical CPU cores: 6
- Logical CPUs: 12
- L2 cache: 7.5 MiB
- L3 cache: 12 MiB
- Available WSL RAM: approximately 7.6 GiB
- Swap: 2 GiB
- OpenVINO version: 2026.4.1
- Available OpenVINO device: CPU only

GPU and NPU inference were not tested because these devices
were not detected by OpenVINO in the current WSL environment.

## 2. Model Storage Requirements

| Model precision | Model directory size |
|---|---:|
| INT4 | 632 MB |
| INT8 | 1.1 GB |
| FP16 | 2.1 GB |

These values represent model storage sizes, not total runtime
memory requirements. Runtime memory also includes KV cache,
temporary tensors, and framework overhead.

## 3. KV Cache Memory Considerations

For TinyLlama, the theoretical FP32 KV cache requirement at
2048 tokens is approximately 88 MiB for batch size 1.

The cache memory grows linearly with sequence length.

INT4 weight quantization does not automatically mean that
the KV cache is also stored in INT4.

## 4. CPU Thread Scaling

TinyLlama INT4 was benchmarked with different CPU thread counts.

| Threads | Average time | Throughput | Busy CPU cores | Peak RAM |
|---:|---:|---:|---:|---:|
| 1 | 8.23 s | 15.68 tok/s | 1.0 | 1367 MB |
| 2 | 5.37 s | 24.03 tok/s | 2.0 | 1368 MB |
| 4 | 3.99 s | 32.30 tok/s | 4.0 | 1367 MB |
| 8 | 4.19 s | 30.80 tok/s | 5.9 | 1368 MB |
| Auto | 4.16 s | 30.98 tok/s | 5.9 | 1368 MB |

Four threads achieved the highest throughput in this experiment.

Increasing threads from 4 to 8 did not improve performance.
This suggests that additional threads provide diminishing
returns, potentially because of shared CPU resource contention.

The four-thread result was also measured separately on Day 5:
31.66 tokens/sec, with 4.07 seconds average generation time
and 1368 MB peak RAM.

## 5. Performance Hint Comparison

| Metric | LATENCY | THROUGHPUT |
|---|---:|---:|
| Generation time | 4.43 s | 4.70 s |
| Throughput | 29.10 tok/s | 27.47 tok/s |
| Busy CPU cores | 6.0 | 3.0 |
| Peak RAM | 1368 MB | 1628 MB |

LATENCY performed slightly better in this single-request
experiment. The main advantage of THROUGHPUT is expected
when handling multiple requests, so a single-request test
cannot establish which hint is best for concurrent serving.

## 6. Batch Inference

| Batch size | Total time | Requests/sec |
|---:|---:|---:|
| 1 | 2.21 s | 0.45 |
| 2 | 2.63 s | 0.76 |
| 4 | 3.60 s | 1.11 |
| 8 | 5.58 s | 1.43 |

Batch size 8 achieved approximately 3.18 times the request
throughput of batch size 1.

Total processing time increased with batch size, but
requests per second also increased.

These results measure request throughput, not token throughput
or individual request latency.

## 7. Memory Bandwidth Experiment

A simple single-thread NumPy experiment estimated approximate
read bandwidth at 16.8 GB/s.

This is an approximate measurement, not a full system memory
bandwidth benchmark or a strict upper bound for LLM inference.

LLM generation performance also depends on CPU computation,
memory hierarchy, threading, cache behavior, and KV cache reads.

## 8. Hardware Utilization Monitoring

During TinyLlama INT4 inference with four threads:

- Several logical CPUs reached approximately 99-100% utilization.
- htop showed approximately 1.16 GB RAM in use during the
  observed run.
- The benchmark measured peak process RAM of 1368 MB.
- Swap usage remained at 0 GB.
- Measured throughput was 31.66 tokens/sec.

This confirms that CPU utilization is an important factor
in local LLM inference performance.

## 9. Bottleneck Analysis and Conclusion

The experiments indicate that CPU resources and memory
behavior influence local LLM inference performance.

Increasing CPU threads from one to four improved throughput
from 15.68 to 32.30 tokens/sec. However, increasing the
thread count further did not improve throughput.

This indicates diminishing returns beyond four threads
in the tested configuration. CPU resource contention and
memory-system limitations are possible contributing factors,
but the experiments do not isolate one definitive bottleneck.

INT4 provided the smallest model footprint and the highest
average throughput in the Week 3 preliminary comparison.

Batch inference improved aggregate request throughput,
while performance hints showed different resource usage
patterns.

Overall, the experiments demonstrate that model precision,
CPU thread count, batching, performance hints, and available
memory all influence local LLM performance.

The results are preliminary measurements from WSL and may
vary with CPU frequency, temperature, system activity, and
runtime configuration. Token counts in the benchmark scripts
were estimated by re-tokenizing generated text.

