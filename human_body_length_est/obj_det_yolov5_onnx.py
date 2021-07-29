import os
import time
from functools import partial

import cv2
import torch
import numpy as np
import onnxruntime

from modules.utils import parse_arguments, DataStreamer
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


def preprocess_image(pil_image, in_size=(640, 640)):
    """preprocesses PIL image and returns a norm np.ndarray
    pil_image = pillow image
    in_size: in_width, in_height
    """
    in_w, in_h = in_size
    resized = letterbox_image(pil_image, (in_w, in_h))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    return img_in


def save_output(detections, image_src, save_path, line_thickness=None, text_bg_alpha=0.0):
    image_src = cv2.cvtColor(image_src, cv2.COLOR_RGB2BGR)
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    if isinstance(image_src, str):
        image_src = cv2.imread(image_src)
    elif isinstance(image_src, np.ndarray):
        image_src = image_src

    h, w = image_src.shape[:2]
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
    cv2.imwrite(save_path, image_src)


@torch.no_grad()
def detect_onnx(src_path,
                media_type,
                threshold=0.6,
                official=True,
                onnx_path="yolov5/yolov5s.onnx",
                output_dir="output",
                num_classes=80):
    session = onnxruntime.InferenceSession(onnx_path)
    model_batch_size = session.get_inputs()[0].shape[0]
    model_h = session.get_inputs()[0].shape[2]
    model_w = session.get_inputs()[0].shape[3]
    in_w = 640 if (model_w is None or isinstance(model_w, str)) else model_w
    in_h = 640 if (model_h is None or isinstance(model_h, str)) else model_h
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    preprocess_func = partial(preprocess_image, in_size=(in_w, in_h))
    data_stream = DataStreamer(src_path, media_type, preprocess_func)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for i, (orig_input, model_input) in enumerate(data_stream):
        batch_size = model_input.shape[0] if isinstance(
            model_batch_size, str) else model_batch_size

        # inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: model_input})

        batch_detections = []
        # model.model[-1].export = boolean ---> True:3 False:4
        if official and len(outputs) == 4:  # recommended
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
            # same anchors for 5s, 5l, 5x
            anchors = [[116, 90, 156, 198, 373, 326], [
                30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

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
        if output_dir is not None:
            save_path = os.path.join(
                output_dir, f"frame_{str(i).zfill(5)}.jpg")
            save_output(batch_detections[0], orig_input, save_path,
                        line_thickness=None, text_bg_alpha=0.0)


if __name__ == '__main__':
    args = parse_arguments("YoloV5 onnx demo")
    t1 = time.time()
    detect_onnx(src_path=args.input_path,
                media_type=args.media_type,
                threshold=args.detection_threshold,
                official=True,  # official yolov5 post-processing
                onnx_path=args.onnx_path,
                output_dir=args.output_dir,
                num_classes=args.num_classes)
    print(
        f"Inference time ({onnxruntime.get_device()}): {time.time() - t1:.2f}s")
