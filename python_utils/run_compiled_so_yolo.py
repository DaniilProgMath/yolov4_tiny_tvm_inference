import tvm
import cv2
import numpy as np
from tvm.contrib import graph_executor
from yolov4_postprocess import post_processing, load_class_names, plot_boxes_cv2

img = cv2.imread("../test_data/frame_15.jpg")
resized = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
img_in = np.expand_dims(img_in, axis=0)
img_in /= 255.0

input_name = "input"
loaded_lib = tvm.runtime.load_module("yolov4_tiny_lib.so")
target = "llvm"
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(loaded_lib["default"](dev))

module.set_input(input_name, img_in)
module.run()
output_boxes_shape = (1, 2535, 1, 4)
output_confs_shape = (1, 2535, 80)
bboxes = module.get_output(0, tvm.nd.empty(output_boxes_shape)).numpy()
confs = module.get_output(1, tvm.nd.empty(output_confs_shape)).numpy()

tvm_output = [bboxes, confs]

boxes = post_processing(0.4, 0.6, tvm_output)
class_names = load_class_names("../config/coco_classes.txt")
plot_boxes_cv2(img, boxes[0], savename='cpp_predictions_tvm_libso.jpg', class_names=class_names)
