import openvino as ov

core=ov.Core()
model=core.read_model("models/tinyllama-int4/openvino_model.xml")

variables=model.get_variables()

print("Total state variables:",len(variables))
print("\nKV Cache Variable Details")
print("-"*70)

for var in sorted(variables,key=lambda v:v.get_info().variable_id):
    info=var.get_info()

    print("\nVariable ID:",info.variable_id)
    print("Data type:",info.data_type)
    print("Data shape:",info.data_shape)