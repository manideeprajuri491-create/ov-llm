import openvino as ov
from collections import Counter

model_path = "models/tinyllama-int4/openvino_model.xml"

core = ov.Core()
model = core.read_model(model_path)

ops = model.get_ordered_ops()
op_counts = Counter(op.get_type_name() for op in ops)

print("=" * 60)
print("INTERNAL MODEL STATE INSPECTION")
print("=" * 60)

print("\nState-related operation counts:")

for op_name in ["ReadValue", "Assign"]:
    print(f"{op_name}: {op_counts.get(op_name, 0)}")

print("\nState-related operations:")
found = False

for op in ops:
    if op.get_type_name() in ["ReadValue", "Assign"]:
        found = True
        print("\nOperation:", op.get_type_name())
        print("Name:", op.get_friendly_name())

        try:
            variable = op.get_variable()
            print("Variable ID:", variable.get_info().variable_id)
        except Exception:
            print("Variable ID: Not available through this API")

if not found:
    print("No ReadValue or Assign operations found.")

print("\nTotal graph operations:", len(ops))
