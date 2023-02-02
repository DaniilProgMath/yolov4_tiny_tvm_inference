import tvm
import onnx
import tvm.relay as relay

target = "llvm"
model_path = "../weights/yolo4-416x416f32.onnx"
input_name = "input"
output_name = "boxes"

onnx_model = onnx.load(model_path)
shape_dict = {input_name: (1, 3, 416, 416)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

lib.export_library("../weights/yolov4_tiny_lib.so")
