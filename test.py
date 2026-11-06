import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("models/tinyllama-int4", "CPU")
prompt = "Explain CPU cache in 3 simple sentences."
result = pipe.generate(prompt, max_new_tokens=80)
print(str(result))