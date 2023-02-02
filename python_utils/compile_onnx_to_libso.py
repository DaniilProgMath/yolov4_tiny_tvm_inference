import tvm
import onnx
import argparse
import tvm.relay as relay


def compile_model(model_path, target):
    input_name = "input"
    onnx_model = onnx.load(model_path)
    shape_dict = {input_name: (1, 3, 416, 416)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    lib.export_library(model_path.replace("onnx", "so"))


parser = argparse.ArgumentParser(
    prog='compile_onnx_to_so',
    description='Скрипт компиляции onnx весов в динамичемкую библиотеку.'
)
parser.add_argument('--model-path', default="../weights/yolo4-416x416f32.onnx")
args = parser.parse_args()

compile_model(args.model_path, "llvm")
