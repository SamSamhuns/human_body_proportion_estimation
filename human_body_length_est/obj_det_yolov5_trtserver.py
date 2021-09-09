from modules.triton_utils import extract_data_from_media, get_client_and_model_metadata_config
from modules.triton_utils import parse_model_grpc, get_inference_responses
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect, plot_one_box
from modules.onnx_utils import letterbox_image, scale_coords, non_max_suppression

from functools import partial
from PIL import Image
import numpy as np
import torch
import time
import cv2
import os

FLAGS = Flag_config()

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


def preprocess(img, width=640, height=640, new_type=np.uint8):
    width = 640 if width is None else width
    height = 640 if height is None else height
    pil_image = Image.fromarray(img).convert('RGB')
    resized = letterbox_image(pil_image, (width, height))
    img_in = np.transpose(resized, (2, 0, 1)).astype(new_type)  # HWC -> CHW
    img_in /= 255.0
    return img_in


def postprocess(results, output_name):
    output = results.as_numpy("output")
    batch_detections = torch.from_numpy(np.array(output))
    batch_detections = non_max_suppression(
        batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)

    detection = batch_detections[0].numpy()
    det_boxes = detection[:, :4]
    det_scores = detection[:, 4]
    det_classes = detection[:, 5]
    return det_boxes, det_scores, det_classes


def run_demo_odet(media_filename,
                  model_name,
                  inference_mode,
                  det_threshold=0.55,
                  save_result_dir=None,  # set to None prevent saving
                  debug=True):
    FLAGS.media_filename = media_filename
    FLAGS.model_name = model_name
    FLAGS.inference_mode = inference_mode
    FLAGS.det_threshold = det_threshold
    FLAGS.result_save_dir = save_result_dir
    FLAGS.model_version = ""  # empty str means use latest
    FLAGS.protocol = "grpc"
    FLAGS.url = '127.0.0.1:8994'
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.debug = debug
    FLAGS.batch_size = 1
    FLAGS.fixed_input_width = None
    FLAGS.fixed_input_height = None
    start_time = time.time()

    if FLAGS.result_save_dir is not None:
        FLAGS.result_save_dir = os.path.join(
            save_result_dir, f"{FLAGS.model_name}")
        os.makedirs(FLAGS.result_save_dir, exist_ok=True)
    if FLAGS.debug:
        print(f"Running model {FLAGS.model_name}")

    model_info = get_client_and_model_metadata_config(FLAGS)
    if model_info == -1:  # error getting model info
        return -1
    triton_client, model_metadata, model_config = model_info

    # input_name, output_name, format, dtype are all lists
    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model_grpc(
        model_metadata, model_config.config)

    # check for dynamic input shapes
    if h == -1:
        h = FLAGS.fixed_input_height
    if w == -1:
        w = FLAGS.fixed_input_width

    filenames = []
    if isinstance(FLAGS.media_filename, str) and os.path.isdir(FLAGS.media_filename):
        filenames = [
            os.path.join(FLAGS.media_filename, f)
            for f in os.listdir(FLAGS.media_filename)
            if os.path.isfile(os.path.join(FLAGS.media_filename, f))
        ]
    else:
        filenames = [
            FLAGS.media_filename,
        ]
    filenames.sort()

    nptype_dict = {"UINT8": np.uint8, "FP32": np.float32, "FP16": np.float16}
    # Important, make sure the first input is the input image
    image_input_idx = 0
    preprocess_dtype = partial(
        preprocess, new_type=nptype_dict[dtype[image_input_idx]])
    # all_reqested_images_orig will be [] if FLAGS.result_save_dir is None
    image_data, all_reqested_images_orig, _, fps = extract_data_from_media(
        FLAGS, preprocess_dtype, filenames, w, h)
    if len(image_data) == 0:
        print("Image data is missing. Aborting inference")
        return -1

    trt_inf_data = (triton_client, input_name,
                    output_name, dtype, max_batch_size)
    # expand the size of the resulting bbox
    x_expand, y_expand = 0, 0
    # if a model with only one input, i.e. edetlite4 is used,
    # the remaining two inputs are ignored
    image_data_list = [image_data,
                       np.array([FLAGS.det_threshold], dtype=np.float32),
                       np.array([x_expand, y_expand], dtype=np.float32)]
    # get inference results
    responses = get_inference_responses(
        image_data_list, FLAGS, trt_inf_data)

    if FLAGS.inference_mode == "video" and FLAGS.result_save_dir is not None:
        vid_writer = cv2.VideoWriter(f"{FLAGS.result_save_dir}/res_video.mp4",
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    counter = 0
    final_result_list = []
    for response in responses:
        det_boxes, det_scores, det_classes = postprocess(response, output_name)
        final_result_list.append([det_boxes, det_scores, det_classes])

        # display boxes on image array
        if FLAGS.result_save_dir is not None:
            drawn_img = all_reqested_images_orig[counter]
            drawn_img = resize_maintaining_aspect(drawn_img, w, h)
            orig_h, orig_w = drawn_img.shape[:2]
            in_h = 640 if h is None else h
            in_w = 640 if w is None else w
            det_boxes[:, :] = scale_coords(
                (in_h, in_w), det_boxes[:, :], (orig_h, orig_w)).round()

            for det_box_xyxy, det_score, det_class in zip(det_boxes, det_scores, det_classes):
                x1, y1 = map(int, det_box_xyxy[:2])
                tl = round(0.004 * (in_h + in_w) / 2) + 1
                label = f"{class_names[int(det_class)]}: {det_score:.2f}"
                cv2.putText(drawn_img, label,
                            (x1 + 3, y1 - 4), 0, tl / 3, [0, 0, 255],
                            thickness=2, lineType=cv2.LINE_AA)
                plot_one_box(det_box_xyxy, drawn_img, color=(255, 0, 0))
            if FLAGS.inference_mode == "image":
                cv2.imwrite(
                    f"{FLAGS.result_save_dir}/frame_{str(counter).zfill(6)}.jpg", drawn_img)
            elif FLAGS.inference_mode == "video":
                vid_writer.write(drawn_img)
        counter += 1
    if FLAGS.debug:
        print(f"Time to process {counter} image(s)={time.time()-start_time}")

    return final_result_list


def main():
    args = parse_arguments("Trt Server Person Detection")
    run_demo_odet(args.input_path,
                  model_name="yolov5m",
                  inference_mode=args.media_type,
                  det_threshold=args.detection_threshold,
                  save_result_dir=args.output_dir,
                  debug=args.debug)


if __name__ == "__main__":
    main()
