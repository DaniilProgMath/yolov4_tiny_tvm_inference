import os
import tvm
import cv2
import sys
import json
import argparse
import numpy as np
from tvm.contrib import graph_executor
from yolov4_postprocess import post_processing, load_class_names, \
    plot_bboxes, filt_classes_from_output, \
    dump_output_to_json, preprocess_image


def load_model(model_path):
    loaded_lib = tvm.runtime.load_module(model_path)
    target = "llvm"
    dev = tvm.device(str(target), 0)
    return graph_executor.GraphModule(loaded_lib["default"](dev))


def inference_model(model, input_data):
    input_name = "input"
    model.set_input(input_name, input_data)
    model.run()
    output_boxes_shape = (1, 2535, 1, 4)
    output_confs_shape = (1, 2535, 80)
    bboxes = model.get_output(0, tvm.nd.empty(output_boxes_shape)).numpy()
    confs = model.get_output(1, tvm.nd.empty(output_confs_shape)).numpy()
    return [bboxes, confs]


def run(model_path, file_path, run_type, visualize_detection=False):
    def make_detection(model, img, timestamp,
                       visualize_detection, visualize_delay):
        img_in = preprocess_image(img)
        output = inference_model(model, img_in)
        output = post_processing(0.4, 0.6, output)
        output, filtered_clases = filt_classes_from_output(
            np.array(output[0]), class_names)
        frame_info = dump_output_to_json(img, output, filtered_clases,
                                         timestamp)

        if visualize_detection:
            res_img = plot_bboxes(frame_info, img)
            cv2.imshow("res_img", res_img)
            cv2.waitKey(visualize_delay)

        return frame_info

    model = load_model(model_path)
    class_names = load_class_names("../config/coco_classes.txt")

    if run_type == "image_inference":
        img = cv2.imread(file_path)
        frame_info = make_detection(model, img, 0, visualize_detection, 0)
        json.dump(frame_info, sys.stdout, indent=4)

    elif run_type == "video_inference":
        cap = cv2.VideoCapture(file_path)
        detections = list()

        while cap.isOpened():
            ret, img = cap.read()
            if img is None:
                break
            frame_info = make_detection(model, img, cap.get(cv2.CAP_PROP_POS_MSEC),
                                        visualize_detection, 5)
            detections.append(frame_info)

        json.dump(detections, sys.stdout, indent=4)


parser = argparse.ArgumentParser(
    prog='run_tvm_yolo',
    description='Решение тестового задания '
                'для компании Flussonic/Эрливидео. '
                'детекция людей и автомобилей '
                'на изображении и видео с '
                'использованием apache tvm.'
)
parser.add_argument('--file-path', default="../test_data/frame_15.jpg")
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

if os.path.isfile(args.file_path):

    file_extension = args.file_path.split(".")[-1]
    if file_extension in ["jpg", "png", "jpeg"]:
        run_type = "image_inference"
    elif file_extension in ["mp4", "avi", "mkv"]:
        run_type = "video_inference"
    else:
        print("Wrong file type.")
        exit()

    run("../weights/yolo4-416x416f32.so",
                args.file_path,
                run_type,
                visualize_detection=args.visualize)
else:
    print(f"File {args.file_path} is not exist")
