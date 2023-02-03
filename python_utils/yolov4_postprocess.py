import cv2
import numpy as np


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    """
    ru:
        Реализация алгоритма не максимального подавления.
    eng:
        Implementation of the non-maximum suppression algorithm.

    :param boxes: List of bbox objects.
    :param confs: List of probabilities.
    :param nms_thresh: non-maximum suppression threshold.
    :return: List of indexes of selected objects.
    """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(conf_thresh, nms_thresh, output):
    """
    ru:
        Функция для обработки результатов выхода сети.
    eng:
        A function to process the results of a network output.

    :param conf_thresh: confidence threshold.
    :param nms_thresh: non-maximum suppression threshold.
    :param output: List of objects in ndarray format (bboxes, scores).
    :return: List of objects in ndarray format (bboxes, scores).
    """
    box_array = output[0]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]
    box_array = box_array[:, :, 0]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0],
                                   ll_box_array[k, 1],
                                   ll_box_array[k, 2],
                                   ll_box_array[k, 3],
                                   ll_max_conf[k],
                                   ll_max_id[k]])
                bboxes = sorted(bboxes, key=lambda x:x[5])[:10]  # Getting top 10 objects

        bboxes_batch.append(bboxes)
    return bboxes_batch


def load_class_names(namesfile):
    """
    ru:
        Функция для загрузки списка классов.
    eng:
        Function to load the list of classes.

    :param namesfile: The filename with list of class.
    :return: list of classes.
    """
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def plot_bboxes(data, frame):
    """
    ru:
        Функция для отрисовки найденных объектов на изображении.
    eng:
        Function for drawing the found objects on the image.

    :param data: Dictionary of detected objects.
    :param frame: ndarray format image.
    :return: ndarray image with boxes and classes object drawn.
    """
    rgb = (255, 255, 255)
    for data_object in data["objects"]:
        x1 = data_object["coords"]["x1"]
        x2 = data_object["coords"]["x2"]
        y1 = data_object["coords"]["y1"]
        y2 = data_object["coords"]["y2"]

        msg = str(data_object["class"]) + " " + str(round(data_object["score"], 3))
        t_size = cv2.getTextSize(msg, 0, 0.7, thickness=2)[0]
        c1, c2 = (x1, y1), (x2, y2)
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

        if data_object["class"] == "person":
            rgb = (0, 255, 0)
        elif data_object["class"] == "car":
            rgb = (255, 0, 0)

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 3)
        frame = cv2.rectangle(frame, (x1, y1), (np.float32(c3[0]),
                                                np.float32(c3[1])), rgb, -1)
        frame = cv2.putText(frame, msg, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2, lineType=cv2.LINE_AA)
    return frame


def filt_classes_from_output(output, class_names):
    """
    ru:
        Функция для фильтрования найденных объектов.
        Возвращает объекты двух классов person и car.

    eng:
        Function for filtering found objects.
        Returns objects of two classes person and car.

    :param output: List of objects in ndarray format (bboxes, scores).
    :param class_names: List of discovered class names.
    :return: List of filtered objects in ndarray format
    (bboxes, scores), list of matching class names
    """

    filtered_output, filtered_clases = list(), list()
    cls_id = np.array(output[:, -1], dtype=np.int)
    detected_clases = [class_names[id] for id in cls_id]

    for i, detected_class in enumerate(detected_clases):
        if detected_class == "person" or detected_class == "car":
            filtered_output.append(output[i])
            filtered_clases.append(detected_class)

    return np.array(filtered_output), filtered_clases


def dump_output_to_json(img, output, class_names, timestamp):
    """
    ru:
        Функция для сохранения результатов детекций в
        словарь, который в последствии можно сохранить в json.

    eng:
        A function for saving detection results to a dictionary,
        which can later be saved as json.

    :param img: ndarray format image.
    :param output: List of objects in ndarray format (bboxes, scores).
    :param class_names: List of discovered class names.
    :param timestamp: List of discovered class names.
    :return: Dictionary of detected objects.
    """

    width = img.shape[1]
    height = img.shape[0]
    frame_info = {"timestamp": timestamp, "objects": []}
    for i, output_i in enumerate(output):
        x1, y1 = int(output_i[0] * width), int(output_i[1] * height)
        x2, y2 = int(output_i[2] * width), int(output_i[3] * height)
        frame_object = {
            "score": output_i[-2],
            "coords": {"x1": x1, "y1": y1,
                       "x2": x2, "y2": y2},
            "class": class_names[i]
        }
        frame_info["objects"].append(frame_object)

    return frame_info


def preprocess_image(img):
    """
    ru:
        Функция выполняющая препроцессинг изображения.

    eng:
        Image preprocessing function.

    :param img: ndarray format image.
    :return: processed image in ndarray format.
    """

    resized = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    return img_in
