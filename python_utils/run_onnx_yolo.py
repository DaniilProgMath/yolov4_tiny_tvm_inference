import cv2
import onnxruntime
import numpy as np
from yolov4_postprocess import post_processing, load_class_names, plot_boxes_cv2

img = cv2.imread("../test_data/frame_15.jpg")
resized = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
img_in = np.expand_dims(img_in, axis=0)
img_in /= 255.0

session = onnxruntime.InferenceSession("../weights/yolo4-416x416f32.onnx", None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run(None, {input_name: img_in})

boxes = post_processing(0.4, 0.6, result)
class_names = load_class_names("../config/coco_classes.txt")

plot_boxes_cv2(img, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)
