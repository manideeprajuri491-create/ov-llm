import openvino as ov

model_path = "models/tinyllama-int4/openvino_model.xml"

core = ov.Core()
model = core.read_model(model_path)

print("=" * 60)
print("TinyLlama OpenVINO Model Inspection")
print("=" * 60)

print("\nMODEL INPUTS")
print("-" * 60)

for port in model.inputs:
    print("Name:", port.get_any_name())
    print("Shape:", port.get_partial_shape())
    print("Type:", port.get_element_type())
    print()

print("\nMODEL OUTPUTS")
print("-" * 60)

for port in model.outputs:
    print("Name:", port.get_any_name())
    print("Shape:", port.get_partial_shape())
    print("Type:", port.get_element_type())
    print()

print("Total inputs:", len(model.inputs))
print("Total outputs:", len(model.outputs))
