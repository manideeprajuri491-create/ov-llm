# TinyLlama Local Inference Optimization with OpenVINO

## Technical Project Report

**Project:** TinyLlama 1.1B Chat – Quantization and CPU Performance Analysis  
**Runtime:** OpenVINO GenAI 2026.4.1  
**Implementation:** C++17, Python benchmarking utilities  
**Execution environment:** Ubuntu on WSL, CPU  
**Processor:** Intel Core i5-12450H (6 physical cores, 12 logical CPUs)  
**Experiment data date:** November 10, 2026

---

## 1. Abstract

This project explores local large language model (LLM) inference using TinyLlama 1.1B Chat and OpenVINO GenAI. The main objectives were to deploy the model on a CPU, compare different weight-precision formats, build a C++ command-line application, and measure the effects of inference thread count and batch size.

Three model variants were evaluated: INT4, INT8, and FP16. Benchmark scripts collected inference time and estimated token throughput. Additional experiments examined thread scaling and aggregate request throughput at different batch sizes.

In the recorded model comparison, INT4 achieved the highest measured throughput, reaching approximately 30.09 tokens/sec in the 300-token-request test. In the thread experiment, 8 threads produced the highest measured throughput, approximately 30.44 tokens/sec. In the batch experiment, batch size 8 achieved approximately 1.30 requests/sec, around 3.13 times the measured throughput of batch size 1.

These results describe the tested WSL CPU environment and experimental runs. They should not be interpreted as universal performance guarantees.

## 2. Objectives

The project was designed to:

1. Run TinyLlama locally using OpenVINO GenAI.
2. compare INT4, INT8, and FP16 model variants.
3. Develop a C++ command-line inference application.
4. Support single-prompt inference and interactive chat.
5. Measure generation performance, including TTFT, TPOT, and throughput.
6. Evaluate CPU thread-count scaling.
7. Evaluate aggregate request throughput at different batch sizes.
8. Export benchmark measurements to CSV and generate charts.
9. Document the implementation, results, and limitations.

## 3. Hardware and Software Environment

| Component | Configuration |
|---|---|
| Processor | Intel Core i5-12450H |
| Physical cores | 6 |
| Logical CPUs | 12 |
| Execution environment | Ubuntu on WSL |
| Available WSL RAM | Approximately 7.6 GiB |
| Swap | Approximately 2 GiB |
| Inference device | CPU |
| OpenVINO GenAI | 2026.4.1 |
| Application language | C++17 |
| Build system | CMake |
| Benchmarking | Python |
| Data format | CSV |
| Visualization | Matplotlib |

GPU and NPU inference were not evaluated. The experiments were conducted using the CPU device.

## 4. Model Variants and Quantization

The project uses TinyLlama 1.1B Chat in three model formats.

| Model format | Approximate model storage |
|---|---:|
| INT4 | 632 MB |
| INT8 | 1.1 GB |
| FP16 | 2.1 GB |

Quantization reduces the precision used to represent model weights. Lower-precision weights generally require less storage and can reduce memory traffic, although actual performance depends on the model, hardware, runtime kernels, and workload.

- **FP16:** Uses 16-bit floating-point weights and serves as the higher-precision comparison in this experiment.
- **INT8:** Uses 8-bit quantized weights, reducing storage compared with FP16.
- **INT4:** Uses 4-bit quantized weights, providing the smallest model storage of the three variants.

The project compares inference performance across these formats. It does not claim that quantization always improves answer quality or that lower precision is suitable for every workload.

## 5. Application Architecture

The C++ application is implemented in `src/main.cpp` and uses OpenVINO GenAI's `ov::genai::LLMPipeline`.

The main execution flow is:

1. Parse command-line arguments.
2. Validate the selected options.
3. Configure OpenVINO runtime properties.
4. Load the selected model and device.
5. Configure text generation.
6. Execute single-prompt inference, interactive chat, or benchmark mode.
7. Display output and performance measurements.

### 5.1 Command-line configuration

The application supports:

| Option | Purpose |
|---|---|
| `--model` | Select the model directory |
| `--prompt` | Set the input prompt |
| `--max-tokens` | Set the maximum number of new tokens |
| `--device` | Select the inference device |
| `--threads` | Set the inference thread count |
| `--perf-hint` | Select `LATENCY` or `THROUGHPUT` |
| `--chat` | Enable interactive conversation |
| `--bench` | Run repeated benchmark iterations |
| `--help` | Display command-line help |

The default model directory is `models/tinyllama-int4`, the default device is CPU, and the default performance hint is LATENCY. A thread count of zero leaves thread selection to the runtime.

The application passes an explicitly selected thread count through `INFERENCE_NUM_THREADS` and configures `PERFORMANCE_HINT` as either `LATENCY` or `THROUGHPUT`.

### 5.2 Generation configuration

The application configures:

- Maximum new tokens using the `--max-tokens` option.
- Repetition penalty: `1.15`.
- Sampling: disabled (`do_sample = false`).

Disabling sampling makes generation deterministic for the configured model and runtime settings, subject to runtime and hardware behavior.

### 5.3 Single-prompt inference

In single-prompt mode, the application sends the selected prompt to `LLMPipeline`, streams the generated response to the terminal, and displays performance metrics.

The streaming callback records the time of the first callback, counts callback invocations, and measures total generation time.

### 5.4 Interactive chat

Chat mode uses `ov::genai::ChatHistory` to retain the system, user, and assistant messages during a conversation.

For each user message:

1. The message is appended to the conversation history.
2. The pipeline generates a response using the history.
3. The assistant response is appended to the history.
4. The next user message is processed with the updated history.

The system prompt requests direct, clear, concise, and accurate answers. Users can enter `exit` or `quit` to leave chat mode.

Conversation history is maintained during the running process; this implementation does not provide persistent conversation storage across separate program executions.

### 5.5 Benchmark mode

The C++ `--bench` mode performs one warm-up generation followed by the requested number of measured iterations. It reports each iteration's elapsed time, total elapsed time, average request time, and requests per second.

This CLI benchmark mode is separate from the Python CSV experiments described below.

## 6. Performance Metrics

The project uses the following performance concepts:

- **Total generation time:** Elapsed time from starting generation until generation completes.
- **TTFT (Time to First Token):** Time from starting generation until the first streaming callback is received.
- **TPOT (Time per Output Token):** Average time per callback after the first callback, calculated by dividing the time after TTFT by the number of remaining callback events.
- **Throughput:** Number of counted callback events divided by total generation time in the C++ streaming display.
- **Requests per second:** Number of completed requests divided by total elapsed time.

The C++ streaming callback counts callback invocations. A callback event is not necessarily identical to a tokenizer token. Therefore, its displayed count and throughput should not automatically be equated with the estimated token counts in the Python CSV benchmark.

The Python model benchmark re-tokenizes decoded output to estimate generated token counts. Those values are explicitly labelled `estimated_generated_tokens`.

## 7. Model Performance Comparison

The following results are from `benchmark_results.csv`.

| Model | Requested tokens | Estimated generated tokens | Time (s) | Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| INT4 | 50 | 51 | 2.1197 | 24.0595 |
| INT4 | 100 | 101 | 3.6538 | 27.6428 |
| INT4 | 300 | 126 | 4.1869 | 30.0942 |
| INT8 | 50 | 51 | 2.6379 | 19.3335 |
| INT8 | 100 | 101 | 4.8194 | 20.9568 |
| INT8 | 300 | 174 | 8.5339 | 20.3892 |
| FP16 | 50 | 51 | 4.7264 | 10.7905 |
| FP16 | 100 | 101 | 8.5855 | 11.7641 |
| FP16 | 300 | 162 | 13.6118 | 11.9014 |

### 7.1 Observations

1. INT4 achieved the highest measured throughput in all three requested-output-size tests.
2. At the 300-token request setting, INT4 measured 30.0942 tokens/sec, INT8 measured 20.3892 tokens/sec, and FP16 measured 11.9014 tokens/sec.
3. INT4's measured throughput in that test was approximately 2.53 times the FP16 result.
4. The 300-token request did not produce 300 estimated tokens for any of the three variants. Generation can stop before the configured maximum when the model emits an end-of-sequence token.
5. The throughput values are specific to these recorded runs. More repeated trials would be needed to establish stable averages and variability.

**Important measurement note:** The requested token count is a maximum generation setting, not a guarantee that the model will generate exactly that many tokens. The estimated output count is based on re-tokenizing decoded text and may differ from the model's internal generation-token count.

## 8. Thread-Scaling Experiment

The thread experiment used the INT4 model with the LATENCY performance hint. Results are from `thread_benchmark_results.csv`.

| Threads | Average time (s) | Throughput (tokens/s) | CPU busy cores | Peak RAM (MB) |
|---:|---:|---:|---:|---:|
| 1 | 10.4743 | 12.3158 | 1.00 | 1364.25 |
| 2 | 5.6681 | 22.7589 | 2.00 | 1364.50 |
| 4 | 4.7971 | 26.8912 | 3.96 | 1364.71 |
| 8 | 4.2373 | 30.4436 | 5.94 | 1365.34 |
| Auto | 4.5996 | 28.0456 | 5.92 | 1368.31 |

### 8.1 Observations

- Increasing the thread count from 1 to 2 and then 4 improved measured throughput.
- In this recorded experiment, 8 threads achieved the highest throughput at 30.4436 tokens/sec.
- Automatic thread selection achieved 28.0456 tokens/sec.
- Peak RAM remained close to 1.36 GB across the tested thread configurations.
- The 8-thread result does not mean that 8 threads are always optimal. Scheduling, workload, runtime settings, CPU contention, and repeated-run variation can change the result.

## 9. Batch-Throughput Experiment

The batch experiment measured aggregate request throughput at different batch sizes using the INT4 model. Results are from `batch_benchmark_results.csv`.

| Batch size | Total time (s) | Requests/sec |
|---:|---:|---:|
| 1 | 2.4028 | 0.4162 |
| 2 | 2.7640 | 0.7236 |
| 4 | 4.1113 | 0.9729 |
| 8 | 6.1399 | 1.3029 |

### 9.1 Observations

- Aggregate request throughput increased as the tested batch size increased.
- Batch size 8 achieved 1.3029 requests/sec.
- Compared with batch size 1, batch size 8 achieved approximately 3.13 times the aggregate request throughput.
- Total elapsed time increased with batch size. This does not contradict the throughput improvement: more requests were processed in each batch.
- Aggregate requests/sec is not the same as individual request latency. The experiment should not be used to claim that every request completes faster at larger batch sizes.

## 10. Memory and KV-Cache Investigation

The project also investigated the model's attention structure and KV-cache requirements.

The inspected TinyLlama configuration uses:

- 22 transformer layers.
- 4 key/value heads.
- Head dimension of 64.
- Grouped-query attention.
- 44 cache state variables: one key and one value state per layer.

For a batch size of 1 and a sequence length of 2048 tokens, the theoretical KV-cache memory estimates are:

| Cache precision | Theoretical memory |
|---|---:|
| FP32 | 88 MiB |
| FP16 | 44 MiB |
| INT8 | 22 MiB |

These are theoretical estimates based on the cache dimensions and element precision. They are not measurements of an implemented quantized KV cache. The project inspected the cache representation but did not implement KV-cache quantization.

PagedAttention was also studied conceptually, but it was not implemented in this project.

## 11. Benchmarking and Visualization

Python scripts were used to run experiments and save measurements as CSV files.

The benchmark data files are:

- `benchmark_results.csv`
- `thread_benchmark_results.csv`
- `batch_benchmark_results.csv`

The chart-generation script is `generate_charts.py`. It produces:

- `charts/model_comparison.png`
- `charts/thread_throughput.png`
- `charts/batch_throughput.png`

The CSV files preserve the recorded measurements, while the charts make it easier to compare model formats, thread configurations, and batch sizes.

### 11.1 Experimental limitations

The results should be interpreted with the following limitations:

1. The CSV data represents recorded runs, not a statistically rigorous repeated-trial study.
2. The Python model benchmark estimates token counts by re-tokenizing decoded text.
3. Generation can stop before reaching the requested maximum token count.
4. Different model formats may produce different output lengths, which affects direct throughput comparisons.
5. WSL CPU scheduling and background processes can affect timing.
6. The thread experiment measures one particular configuration and run; results may vary on other machines.
7. Batch throughput does not directly describe per-request latency.
8. GPU and NPU inference were not tested.
9. KV-cache precision figures are theoretical, not results from an implemented cache-quantization feature.
10. The model's answer quality was explored with a small set of prompts, not a comprehensive benchmark.

## 12. Challenges and Lessons Learned

The project involved more than simply loading a model. Important engineering tasks included:

- Configuring the OpenVINO GenAI runtime and its shared-library dependencies.
- Building and linking a C++ application with CMake.
- Handling model-directory and generation configuration options.
- Implementing streaming output and measuring callback timing.
- Maintaining conversation history in interactive chat mode.
- Separating warm-up time from measured iterations in the C++ benchmark.
- Exporting benchmark results to CSV and generating charts.
- Understanding the difference between model-weight quantization and KV-cache memory.
- Interpreting throughput and latency as different performance measures.

The experiments also demonstrated that model size, thread count, and batch size affect different aspects of local inference performance.

## 13. Future Improvements

Potential next steps include:

1. Run multiple trials for every configuration and report mean, median, and standard deviation.
2. Use exact generation-token counts from runtime instrumentation where available.
3. Compare LATENCY and THROUGHPUT hints under both single-request and concurrent workloads.
4. Measure per-request latency and tail latency during batch or concurrent inference.
5. Evaluate GPU and NPU execution on supported hardware.
6. Investigate KV-cache memory optimization and quantization.
7. Evaluate model answer quality with a larger, repeatable test set.
8. Add automated benchmark validation and chart regeneration to the project workflow.
9. Improve chat usability by removing development-only debug output and adding optional conversation export.

## 14. Conclusion

This project successfully deployed TinyLlama 1.1B Chat locally with OpenVINO GenAI and developed a C++ command-line application supporting single-prompt inference, interactive chat, streaming output, and repeated-request benchmarking.

The recorded CPU experiments showed that INT4 delivered the highest throughput among the tested model formats. Thread scaling improved throughput in the tested range, with 8 threads producing the highest measured result in that run. Increasing batch size improved aggregate request throughput, although total batch time also increased.

The project provided practical experience with LLM deployment, quantization, C++ integration, CPU performance analysis, benchmark automation, and technical documentation. The measurements are specific to the tested environment and should be extended with repeated trials and broader quality and hardware evaluations.

## 15. Project Files

Important project files include:

| File or directory | Purpose |
|---|---|
| `src/main.cpp` | C++ inference application |
| `CMakeLists.txt` | CMake build configuration |
| `models/` | INT4, INT8, and FP16 model directories |
| `bench.py` | Model-format benchmark |
| `thread_bench.py` | Thread-scaling experiment |
| `batch_bench.py` | Batch-throughput experiment |
| `generate_charts.py` | Chart generation |
| `benchmark_results.csv` | Model benchmark measurements |
| `thread_benchmark_results.csv` | Thread benchmark measurements |
| `batch_benchmark_results.csv` | Batch benchmark measurements |
| `charts/` | Generated benchmark charts |
| `README.md` | Project setup and usage documentation |
| `notes.md` | Development notes and investigation log |

---

**End of report**
