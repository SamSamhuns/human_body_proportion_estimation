from modules.triton_utils import parse_model_grpc, get_client_and_model_metadata_config
from modules.triton_utils import extract_data_from_media, get_inference_responses
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect, plot_one_box

import numpy as np
import time
import cv2
import os


FLAGS = Flag_config()


def preprocess(img, width=640, height=480, new_type=np.uint8):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_maintaining_aspect(img, width, height).astype(new_type)
    return img


def postprocess(results, output_name):
    output_set = set(output_name)

    if "output_0" in output_set and "output_0" in output_set and "output_0" in output_set:
        det_boxes = results.as_numpy("output_0")[0]
        det_scores = results.as_numpy("output_1")[0]
        det_classes = results.as_numpy("output_2")[0]
    elif "detection_boxes" in output_set and "detection_classes" in output_set and "detection_scores" in output_set:
        det_boxes = results.as_numpy("detection_boxes")
        det_scores = results.as_numpy("detection_scores")
        det_classes = results.as_numpy("detection_classes")

    det_scores = [score for score in det_scores if score > FLAGS.det_threshold]
    det_boxes = det_boxes[:len(det_scores)]
    det_classes = det_classes[:len(det_scores)]
    return det_boxes, det_scores, det_classes


def run_demo_odet(media_filename,
                  model_name,
                  inference_mode,
                  det_threshold=0.55,
                  save_result_dir=None):  # set to None prevent saving
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
    FLAGS.debug = False
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

    max_batch_size, input_name, output_name, h, w, c, format, dtype = parse_model_grpc(
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

    # all_reqested_images_orig will be [] if FLAGS.result_save_dir is None
    image_data, all_reqested_images_orig, fps = extract_data_from_media(
        FLAGS, preprocess, filenames, w, h)

    if len(image_data) == 0:
        print("Image data is missing. Aborting inference")
        return -1

    trt_inf_data = (triton_client, input_name,
                    output_name, dtype, max_batch_size)
    # get inference results
    responses = get_inference_responses(image_data, FLAGS, trt_inf_data)

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
            if len(det_boxes) > 0 and np.amax(det_boxes) <= 1.0:
                hscale = drawn_img.shape[0]
                wscale = drawn_img.shape[1]
            else:
                hscale = 1
                wscale = 1

            for box_yxyx in det_boxes:
                # change orientation of bounding box coords
                y1, x1, y2, x2 = box_yxyx
                box_xyxy = (x1, y1, x2, y2)
                # checking the output scales
                plot_one_box(box_xyxy, drawn_img, wscale=wscale,
                             hscale=hscale, color=(255, 0, 0))
            if FLAGS.result_save_dir is not None:
                if FLAGS.inference_mode == "image":
                    cv2.imwrite(
                        f"{FLAGS.result_save_dir}/frame_{counter}.jpg", drawn_img)
                elif FLAGS.inference_mode == "video":
                    vid_writer.write(drawn_img)
        counter += 1
    if FLAGS.debug:
        print(f"Time to process {counter} image(s)={time.time()-start_time}")

    return final_result_list


def main():
    args = parse_arguments("Person Detection")
    run_demo_odet(args.input_path,
                  model_name="edetlite4_modified",
                  inference_mode=args.media_type,
                  det_threshold=args.detection_threshold,
                  save_result_dir=args.output_dir)


if __name__ == "__main__":
    main()
