
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/openvino.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cctype>

using Clock = std::chrono::steady_clock;

// --------------------------------------------------
// Command-line options
// --------------------------------------------------

struct Options {
    std::string model = "models/tinyllama-int4";
    std::string prompt = "What is an operating system?";
    std::string device = "CPU";
    std::string perf_hint = "LATENCY";

    size_t max_tokens = 100;
    int threads = 0;
    int bench_iterations = 5;

    bool chat = false;
    bool help = false;
};

// --------------------------------------------------
// Display help
// --------------------------------------------------

void print_help() {
    std::cout
        << "\nTinyLlama OpenVINO CLI\n\n"
        << "Usage:\n"
        << "  ./build/ov_llm_cli [options]\n\n"
        << "Options:\n"
        << "  --model <path>          Model directory\n"
        << "  --prompt <text>         Input prompt\n"
        << "  --max-tokens <number>   Maximum generated tokens\n"
        << "  --device <device>       Device (CPU)\n"
        << "  --threads <number>      Number of inference threads\n"
        << "  --perf-hint <hint>      LATENCY or THROUGHPUT\n"
        << "  --chat                  Enable interactive chat\n"
        << "  --bench <iterations>    Run benchmark\n"
        << "  --help                  Display help\n\n";
}

// --------------------------------------------------
// Parse command-line arguments
// --------------------------------------------------

Options parse_arguments(int argc, char* argv[]) {
    Options options;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        auto require_value = [&](const std::string& name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "Missing value for " + name
                );
            }
        };

        if (arg == "--model") {
            require_value(arg);
            options.model = argv[++i];
        }
        else if (arg == "--prompt") {
            require_value(arg);
            options.prompt = argv[++i];
        }
        else if (arg == "--max-tokens") {
            require_value(arg);
            options.max_tokens = std::stoul(argv[++i]);
        }
        else if (arg == "--device") {
            require_value(arg);
            options.device = argv[++i];
        }
        else if (arg == "--threads") {
            require_value(arg);
            options.threads = std::stoi(argv[++i]);
        }
        else if (arg == "--perf-hint") {
            require_value(arg);
            options.perf_hint = argv[++i];

            std::transform(
                options.perf_hint.begin(),
                options.perf_hint.end(),
                options.perf_hint.begin(),
                [](unsigned char c) {
                    return static_cast<char>(std::toupper(c));
                }
            );

            if (options.perf_hint != "LATENCY" &&
                options.perf_hint != "THROUGHPUT") {
                throw std::runtime_error(
                    "Invalid performance hint. Use LATENCY or THROUGHPUT."
                );
            }
        }
        else if (arg == "--chat") {
            options.chat = true;
        }
        else if (arg == "--bench") {
            require_value(arg);
            options.bench_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--help") {
            options.help = true;
        }
        else {
            throw std::runtime_error(
                "Unknown argument: " + arg
            );
        }
    }

    return options;
}

// --------------------------------------------------
// Main function
// --------------------------------------------------

int main(int argc, char* argv[]) {

    try {

        // Parse command-line arguments
        Options options = parse_arguments(argc, argv);

        if (options.help) {
            print_help();
            return 0;
        }

        // Validate options
        if (options.max_tokens == 0) {
            throw std::runtime_error(
                "Maximum tokens must be greater than zero."
            );
        }

        if (options.threads < 0) {
            throw std::runtime_error(
                "Thread count cannot be negative."
            );
        }

        if (options.bench_iterations <= 0) {
            throw std::runtime_error(
                "Benchmark iterations must be greater than zero."
            );
        }

        // --------------------------------------------------
        // Runtime properties
        // --------------------------------------------------

        ov::AnyMap properties;

        if (options.threads > 0) {
            properties["INFERENCE_NUM_THREADS"] = options.threads;
        }

        if (options.perf_hint == "LATENCY") {
            properties["PERFORMANCE_HINT"] = "LATENCY";
        }
        else {
            properties["PERFORMANCE_HINT"] = "THROUGHPUT";
        }

        // --------------------------------------------------
        // Load model
        // --------------------------------------------------

        std::cout << "Loading model: "
                  << options.model << std::endl;

        std::cout << "Device: "
                  << options.device << std::endl;

        std::cout << "Performance hint: "
                  << options.perf_hint << std::endl;

        ov::genai::LLMPipeline pipe(
            options.model,
            options.device,
            properties
        );

        std::cout << "Model loaded successfully!\n";

        // --------------------------------------------------
        // Generation configuration
        // --------------------------------------------------

        ov::genai::GenerationConfig config;

        config.max_new_tokens = options.max_tokens;
        config.repetition_penalty = 1.15f;
        config.do_sample = false;

        // --------------------------------------------------
        // Streaming callback and performance measurements
        // --------------------------------------------------

        auto run_streaming = [&](auto&& generate_function) {

            size_t generated_tokens = 0;

            auto start_time = Clock::now();
            auto first_token_time = start_time;

            bool first_token_received = false;

            auto streamer = [&](std::string token)
                -> ov::genai::StreamingStatus {

                if (!first_token_received) {
                    first_token_time = Clock::now();
                    first_token_received = true;
                }

                generated_tokens++;

                std::cout << token << std::flush;

                return ov::genai::StreamingStatus::RUNNING;
            };

            auto result = generate_function(streamer);

            auto end_time = Clock::now();

            double total_time =
                std::chrono::duration<double>(
                    end_time - start_time
                ).count();

            double ttft = 0.0;

            if (first_token_received) {
                ttft =
                    std::chrono::duration<double>(
                        first_token_time - start_time
                    ).count();
            }

            double tpot = 0.0;

            if (generated_tokens > 1) {
                tpot =
                    (total_time - ttft) /
                    (generated_tokens - 1);
            }

            double throughput = 0.0;

            if (total_time > 0.0) {
                throughput =
                    generated_tokens / total_time;
            }

            std::cout << "\n\n";

            std::cout << std::fixed
                      << std::setprecision(4);

            std::cout << "Performance Metrics\n";
            std::cout << "------------------------------\n";

            std::cout << "Generated tokens: "
                      << generated_tokens << "\n";

            std::cout << "Total generation time: "
                      << total_time << " seconds\n";

            std::cout << "TTFT: "
                      << ttft << " seconds\n";

            std::cout << "TPOT: "
                      << tpot << " seconds\n";

            std::cout << "Throughput: "
                      << throughput << " tokens/sec\n";

            return result;
        };

        // --------------------------------------------------
        // Interactive chat mode
        // --------------------------------------------------

        if (options.chat) {

            std::cout << "\n";
            std::cout << "TinyLlama Interactive Chat\n";
            std::cout << "Type 'exit' to quit.\n\n";

            ov::genai::ChatHistory history;

            // Updated concise system prompt
            const std::string system_prompt =
                "You are a helpful and accurate assistant. "
                "Answer the user's question directly. "
                "Follow the requested format exactly. "
                "If asked for three bullet points, provide exactly three bullet points. "
                "Each bullet point must contain only one sentence when requested. "
                "Do not add introductions or conclusions unless requested. "
                "Use simple and clear language. "
                "Do not include unrelated information. "
                "Do not invent facts. "
                "If you are unsure, say so.";

            history.push_back({
                {"role", "system"},
                {"content", system_prompt}
            });

            std::string user_input;

            while (true) {

                std::cout << "\nYou: ";

                if (!std::getline(std::cin, user_input)) {
                    break;
                }

                if (user_input.empty()) {
                    continue;
                }

                if (user_input == "exit" ||
                    user_input == "quit") {
                    std::cout << "Exiting chat...\n";
                    break;
                }

                // Add user message to conversation history
                history.push_back({
                    {"role", "user"},
                    {"content", user_input}
                });

                std::cout << "\nTinyLlama: ";

                // Generate response using conversation history
                auto result = run_streaming(
                    [&](auto& streamer) {
                        return pipe.generate(
                            history,
                            config,
                            streamer
                        );
                    }
                );

                // Save assistant response to history
                if (!result.texts.empty()) {

                    history.push_back({
                        {"role", "assistant"},
                        {"content", result.texts.at(0)}
                    });
                }

            }

            return 0;
        }

        // --------------------------------------------------
        // Benchmark mode
        // --------------------------------------------------

        if (argc > 1) {

            bool benchmark_requested = false;

            for (int i = 1; i < argc; i++) {
                if (std::string(argv[i]) == "--bench") {
                    benchmark_requested = true;
                    break;
                }
            }

            if (benchmark_requested) {

                std::cout << "\nBenchmark Mode\n";
                std::cout << "Prompt: "
                          << options.prompt << "\n";

                std::cout << "Iterations: "
                          << options.bench_iterations << "\n\n";

                std::cout << "Warming up...\n";

                // Warm-up inference
                pipe.generate(
                    options.prompt,
                    config
                );

                double total_time = 0.0;

                for (int i = 0;
                     i < options.bench_iterations;
                     i++) {

                    auto start = Clock::now();

                    pipe.generate(
                        options.prompt,
                        config
                    );

                    auto end = Clock::now();

                    double elapsed =
                        std::chrono::duration<double>(
                            end - start
                        ).count();

                    total_time += elapsed;

                    std::cout << "Iteration "
                              << (i + 1)
                              << ": "
                              << elapsed
                              << " seconds\n";
                }

                double average_time =
                    total_time / options.bench_iterations;

                double requests_per_second =
                    options.bench_iterations / total_time;

                std::cout << "\nBenchmark Results\n";
                std::cout << "------------------------------\n";

                std::cout << "Total iterations: "
                          << options.bench_iterations << "\n";

                std::cout << "Total time: "
                          << total_time << " seconds\n";

                std::cout << "Average request time: "
                          << average_time << " seconds\n";

                std::cout << "Requests per second: "
                          << requests_per_second << "\n";

                return 0;
            }
        }

        // --------------------------------------------------
        // Single prompt mode
        // --------------------------------------------------

        std::cout << "\nPrompt: "
                  << options.prompt << "\n\n";

        std::cout << "TinyLlama: ";

        run_streaming(
            [&](auto& streamer) {
                return pipe.generate(
                    options.prompt,
                    config,
                    streamer
                );
            }
        );

    }
    catch (const std::exception& e) {

        std::cerr << "\nError: "
                  << e.what() << std::endl;

        return 1;
    }

    return 0;
}
