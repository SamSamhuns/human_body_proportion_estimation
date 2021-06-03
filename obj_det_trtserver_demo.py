from modules.triton_utils import parse_model_grpc, get_client_and_model_metadata_config
from modules.triton_utils import requestGenerator, extract_data_from_media
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect, plot_one_box

from tritonclient.utils import InferenceServerException
import numpy as np
import subprocess
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
                  inference_mode='video',
                  det_threshold=0.55,
                  save_result_dir=None):  # set to None prevent saving
    FLAGS.media_filename = media_filename
    FLAGS.model_name = model_name
    FLAGS.inference_mode = inference_mode
    FLAGS.det_threshold = det_threshold
    FLAGS.frames_save_dir = save_result_dir
    FLAGS.model_version = ""
    FLAGS.protocol = "grpc"
    FLAGS.url = '127.0.0.1:8994'
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.debug = False
    FLAGS.batch_size = 1
    FLAGS.fixed_input_width = 512
    FLAGS.fixed_input_height = 512
    start_time = time.time()

    if FLAGS.frames_save_dir is not None:
        FLAGS.frames_save_dir = save_result_dir + f"_{FLAGS.model_name}"
        os.makedirs(FLAGS.frames_save_dir, exist_ok=True)
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
    w, h = None, None

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

    # all_reqested_images_orig will be [] if FLAGS.frames_save_dir is None
    (image_data,
     all_reqested_images_orig,
     fps, fmt, leading_zeros) = extract_data_from_media(FLAGS, preprocess, filenames, w, h)

    if len(image_data) == 0:
        print("Image data was missing")
        return -1

    responses = []
    image_idx = 0
    last_request = False
    sent_count = 0

    while not last_request:
        repeated_image_data = []

        for idx in range(FLAGS.batch_size):
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True
        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]
        if max_batch_size == 0:
            batched_image_data = np.expand_dims(batched_image_data, 0)
        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    batched_image_data, input_name, output_name, dtype, FLAGS):
                sent_count += 1
                responses.append(
                    triton_client.infer(FLAGS.model_name,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=FLAGS.model_version,
                                        outputs=outputs))

        except InferenceServerException as e:
            print("inference failed: " + str(e))
            return -1

    counter = 0
    file = filenames[counter]
    final_result_list = []
    if FLAGS.debug:
        print(file)

    for response in responses:
        det_boxes, det_scores, det_classes = postprocess(response, output_name)
        final_result_list.append([det_boxes, det_scores, det_classes])

        # display boxes on image array
        if FLAGS.frames_save_dir is not None:
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
                plot_one_box(box_xyxy,
                             drawn_img,
                             wscale=wscale,
                             hscale=hscale,
                             color=(255, 0, 0))

            if FLAGS.debug:
                print(f"frame_{counter}.jpg")
            if FLAGS.frames_save_dir:
                cv2.imwrite(os.path.join(FLAGS.frames_save_dir, f"frame_%{fmt}.jpg" % (counter)),
                            drawn_img)

        counter += 1
    if FLAGS.debug:
        print(f"Time to process {counter} images={time.time()-start_time}")

    # cur_pred.mask_results = mask_results
    if FLAGS.inference_mode == "video" and FLAGS.frames_save_dir is not None:
        command = ["ffmpeg",
                   "-r", f"{fps}",
                   "-start_number", "0" * (leading_zeros),
                   "-i", f"{FLAGS.frames_save_dir}/frame_%{leading_zeros}d.jpg",
                   "-vcodec", "libx264",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-y", "-an", f"{FLAGS.frames_save_dir}/res_video.mp4"]
        output, error = subprocess.Popen(
            command, universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
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
