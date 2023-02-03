import cv2
import argparse
import onnxruntime
import numpy as np
from yolov4_postprocess import post_processing, load_class_names, \
    plot_bboxes, filt_classes_from_output, \
    dump_output_to_json, preprocess_image


def run_onnx_inference(image_path):
    """
    ru:
        Функция выполняющая инференс детектора
        и визуализацию найденных объектов.

    :param image_path: Путь к изображению.
    :return: None

    eng:
        A function that performs detector inference
        and visualization of found objects.

    :param image_path: Path to the image.
    :return: None
    """

    img = cv2.imread(image_path)
    img_in = preprocess_image(img)

    session = onnxruntime.InferenceSession("../weights/yolo4-416x416f32.onnx", None)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img_in})

    boxes = post_processing(0.4, 0.6, result)
    class_names = load_class_names("../config/coco_classes.txt")

    output, filtered_clases = filt_classes_from_output(
        np.array(boxes[0]), class_names)
    frame_info = dump_output_to_json(img, output, filtered_clases, 0)
    res_img = plot_bboxes(frame_info, img)
    cv2.imshow("res_img", res_img)
    cv2.waitKey(0)


parser = argparse.ArgumentParser(
    prog='run_onnx_yolo',
    description='Скрипт инференса через onnxruntime'
)
parser.add_argument('--image-path', default="../test_data/frame_15.jpg")
args = parser.parse_args()
run_onnx_inference(args.image_path)
