import os
import cv2
import time
import glob
import numpy as np
import onnxruntime

import torch
from PIL import Image
from modules.utils import parse_arguments
from modules.onnx_utils import letterbox_image, scale_coords, non_max_suppression, w_non_max_suppression

# class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def preprocess_image(pil_image, in_size):
    """preprocesses PIL image and returns a norm np.ndarray
    """
    in_w, in_h = in_size
    resized = letterbox_image(pil_image, (in_w, in_h))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    return img_in


def get_input_batch(src_path, in_size, media_type='image'):
    allowed_modes = {'image', 'video'}
    if media_type not in allowed_modes:
        raise ValueError(
            f"{media_type} is not allowed. Only {allowed_modes} are allowed")

    w, h = in_size
    if media_type == 'image':
        input_batch = np.expand_dims(
            preprocess_image(Image.open(src_path), (w, h)), axis=0)
    elif media_type == 'video':
        cap = cv2.VideoCapture(src_path)
        input_batch = []
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                input_batch.append(preprocess_image(
                    Image.fromarray(frame), w, h))
            else:
                break
            i += 1
        cap.release()
        input_batch = np.array(input_batch)

    return input_batch


def detect_onnx(image_path, media_type, official=True, onnx_path="models/yolov5s/1/model.onnx", num_classes=80):
    session = onnxruntime.InferenceSession(onnx_path)
    model_batch_size = session.get_inputs()[0].shape[0]
    model_h = session.get_inputs()[0].shape[2]
    model_w = session.get_inputs()[0].shape[3]
    in_w = 640 if (model_w is None or isinstance(model_w, str)) else model_w
    in_h = 640 if (model_h is None or isinstance(model_h, str)) else model_h
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    model_input = get_input_batch(image_path, (in_w, in_h))
    batch_size = model_input.shape[0] if isinstance(
        model_batch_size, str) else model_batch_size

    # inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: model_input})

    batch_detections = []
    # model.model[-1].export = boolean ---> True:3 False:4
    if official and len(outputs) == 4:
        # model.model[-1].export = False ---> outputs[0] (1, xxxx, 85)
        # Use the official code directly
        batch_detections = torch.from_numpy(np.array(outputs[0]))
        batch_detections = non_max_suppression(
            batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    else:
        # model.model[-1].export = False ---> outputs[1]/outputs[2]/outputs[2]
        # model.model[-1].export = True  ---> outputs
        # (1, 3, 20, 20, 85)
        # (1, 3, 40, 40, 85)
        # (1, 3, 80, 80, 85)
        # Handwriting by yourself (part of the principle comes from yolo.py Detect)
        anchors = [[116, 90, 156, 198, 373, 326], [
            30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]  # 5s, 5l, 5x

        boxs = []
        a = torch.tensor(anchors).float().view(3, -1, 2)
        anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
        if len(outputs) == 4:
            outputs = [outputs[1], outputs[2], outputs[3]]
        for index, out in enumerate(outputs):
            out = torch.from_numpy(out)
            # batch = out.shape[1]
            feature_w = out.shape[2]
            feature_h = out.shape[3]

            # Feature map corresponds to the original image zoom factor
            stride_w = int(in_w / feature_w)
            stride_h = int(in_h / feature_h)

            grid_x, grid_y = np.meshgrid(
                np.arange(feature_w), np.arange(feature_h))

            # cx, cy, w, h
            pred_boxes = torch.FloatTensor(out[..., :4].shape)
            pred_boxes[..., 0] = (torch.sigmoid(
                out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
            pred_boxes[..., 1] = (torch.sigmoid(
                out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
            pred_boxes[..., 2:4] = (torch.sigmoid(
                out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh

            conf = torch.sigmoid(out[..., 4])
            pred_cls = torch.sigmoid(out[..., 5:])

            output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                                conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, num_classes)),
                               -1)
            boxs.append(output)

        outputx = torch.cat(boxs, 1)
        # NMS
        batch_detections = w_non_max_suppression(
            outputx, num_classes, conf_thres=0.4, nms_thres=0.3)

    return batch_detections


def save_output(detections, image_path, output_dir, line_thickness=None, text_bg_alpha=0.0):
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    image_src = cv2.imread(image_path)
    h, w = image_src.shape[:2]
    # resized = np.array(image_src)
    # resized = letterbox_image(image_src, (640, 640))
    # resized = np.array(resized)
    boxs[:, :] = scale_coords((640, 640), boxs[:, :], (h, w)).round()

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = map(int, box)
        np.random.seed(int(labels[i].numpy()) + 2020)
        color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
        print(x1, y1, x2, y2)
        cv2.rectangle(image_src, (x1, y1), (x2, y2), color, thickness=max(
            int((w + h) / 600), 1), lineType=cv2.LINE_AA)
        label = '%s %.2f' % (class_names[int(labels[i].numpy())], confs[i])
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        if text_bg_alpha == 0.0:
            cv2.rectangle(image_src, (x1 - 1, y1), c2,
                          color, cv2.FILLED, cv2.LINE_AA)
        else:
            # Transparent text background
            alphaReserve = text_bg_alpha  # 0: opaque 1: transparent
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
            xMax, yMax = int(x1 + t_size[0]), int(y1)
            image_src[yMin:yMax, xMin:xMax, 0] = image_src[yMin:yMax,
                                                           xMin:xMax, 0] * alphaReserve + BChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 1] = image_src[yMin:yMax,
                                                           xMin:xMax, 1] * alphaReserve + GChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 2] = image_src[yMin:yMax,
                                                           xMin:xMax, 2] * alphaReserve + RChannel * (1 - alphaReserve)
        cv2.putText(image_src, label, (x1 + 3, y1 - 4), 0, tl / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)
        print(box.numpy(), confs[i].numpy(),
              class_names[int(labels[i].numpy())])

    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, image_name)
    cv2.imwrite(save_path, image_src)


if __name__ == '__main__':
    args = parse_arguments("YoloV5 onnx demo")
    with torch.no_grad():
        t1 = time.time()
        detections = detect_onnx(
            image_path=args.input_path, media_type=args.media_type, official=True)
        print(
            f"Inference time ({onnxruntime.get_device()}): {time.time() - t1:.2f}s")
        print("Detections are: ", detections[0])
        if detections[0] is not None:
            save_output(detections[0], args.input_path,
                        output_dir=args.output_dir, text_bg_alpha=0.6)
