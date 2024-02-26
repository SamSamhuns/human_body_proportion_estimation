from functools import partial
import time
import os

from modules.triton_utils import parse_model_grpc, get_client_and_model_metadata_config
from modules.triton_utils import get_inference_responses, extract_data_from_media
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect, plot_one_box
from modules.pose_estimator import PoseEstimator

import numpy as np
import cv2
FLAGS = Flag_config()


def preprocess(img, width=640, height=480, new_type=np.uint8):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_maintaining_aspect(img, width, height).astype(new_type)
    return img


def postprocess(results, output_name):
    boxes = results.as_numpy("ENSEMBLE_OUTPUT_FILTER_DET_BOXES")
    heatmaps = results.as_numpy("ENSEMBLE_OUTPUT_HEATMAPS")
    if len(heatmaps.shape) == 3:
        heatmaps = np.expand_dims(heatmaps, axis=0)
    return boxes.copy(), heatmaps


def run_pdet_pose(media_filename,
                  model_name="ensemble_edet4_person_det_pose",
                  # list with input heights, def avg male height
                  person_height=[175],
                  inference_mode='image',
                  det_threshold=0.70,
                  # set to None prevent saving
                  save_result_dir=None,
                  grpc_port='8994',
                  debug=True):
    FLAGS.media_filename = media_filename
    FLAGS.model_name = model_name
    FLAGS.p_height = person_height
    FLAGS.inference_mode = inference_mode
    FLAGS.det_threshold = det_threshold
    FLAGS.result_save_dir = save_result_dir
    FLAGS.model_version = ""
    FLAGS.protocol = "grpc"
    FLAGS.url = f'127.0.0.1:{grpc_port}'
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.debug = debug
    FLAGS.batch_size = 1
    # if model in_size are dynamic and below set to None, no resizing of original input is done
    FLAGS.fixed_input_width = None
    FLAGS.fixed_input_height = None
    # nose, reye, leye, rear, lear, rshoulder,
    # lshoulder, relbow, lelbow, rwrist, lwrist,
    # rhip, lhip, rknee, lknee, rankle, lankle
    FLAGS.KEYPOINT_THRES_LIST = [0.45, 0.46, 0.45, 0.40, 0.34, 0.10, 0.10, 0.10, 0.10,
                                 0.24, 0.30, 0.11, 0.10, 0.15, 0.10, 0.25, 0.20]
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

    nptype_dict = {"UINT8": np.uint8, "FP32": np.float32, "FP16": np.float16}
    # Important, make sure the first input is the input image
    image_input_idx = 0
    preprocess_dtype = partial(
        preprocess, new_type=nptype_dict[dtype[image_input_idx]])
    # all_req_imgs_orig will be [] if FLAGS.result_save_dir is None
    image_data, all_req_imgs_orig, all_req_imgs_orig_size, fps = extract_data_from_media(
        FLAGS, preprocess_dtype, filenames, w, h)

    if len(image_data) == 0:
        print("Image data was missing")
        return []

    trt_inf_data = (triton_client, input_name,
                    output_name, dtype, max_batch_size)
    # expand the coords of the bbox to send more info to pose extractor
    x_expand = w // 17 if w is not None else image_data[0].shape[0] // 17
    y_expand = 0
    image_data_list = [image_data,
                       np.array([FLAGS.det_threshold], dtype=np.float32),
                       np.array([x_expand, y_expand], dtype=np.float32)]
    # get inference results
    responses = get_inference_responses(image_data_list, FLAGS, trt_inf_data)

    if FLAGS.inference_mode == "video" and FLAGS.result_save_dir is not None:
        _, vh, vw, _ = all_req_imgs_orig_size
        vid_writer = cv2.VideoWriter(f"{FLAGS.result_save_dir}/res_video.mp4",
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw, vh))

    counter = 0
    final_result_list = []
    for response in responses:
        boxes, heatmaps = postprocess(response, output_name)
        final_result_list.append([boxes, heatmaps])

        # display boxes on image array
        if FLAGS.result_save_dir is not None:
            drawn_img = all_req_imgs_orig[counter]
            drawn_img = resize_maintaining_aspect(drawn_img, w, h)
            h, w, c = drawn_img.shape
        else:
            _, h, w, c = all_req_imgs_orig_size

        # de-normalize bbox coords
        boxes *= [h, w, h, w]
        # for each human detected/crop/heatmap
        cmaps = [(255, 255, 0), (0, 0, 255)]
        for i, (heatmap, box) in enumerate(zip(heatmaps, boxes)):
            keypts, keypts_score = PoseEstimator.get_max_pred_keypts_from_heatmap(
                heatmap)
            x1, y1 = int(box[1]), int(box[0])  # top left
            x2, y2 = int(box[3]), int(box[2])  # bottom right

            # change coord axes of keypts to match that of orig image
            _, hmap_height, hmap_width = heatmap.shape
            crop_width, crop_height = x2 - x1, y2 - y1
            # following is equi to this: x, y = int((x / hw) * cw) + x1, int((y / hh) * ch) + y1
            keypts /= [hmap_width, hmap_height]
            keypts *= [crop_width, crop_height]
            keypts += [x1, y1]

            ig_kp_idx = {i for i, score in enumerate(keypts_score)
                         if score < FLAGS.KEYPOINT_THRES_LIST[i]}

            # get estimations of body part lengths
            height_pixel = y2 - y1
            height_cm = FLAGS.p_height[min(i, len(FLAGS.p_height) - 1)]
            pixel_to_cm = height_cm / height_pixel
            dist_dict = PoseEstimator.get_keypoint_dist_dict(
                pixel_to_cm, keypts, ignored_kp_idx=ig_kp_idx)
            final_result_list[-1].append(dist_dict)

            if FLAGS.result_save_dir is not None:
                # uncomment to plot bounding boxes
                plot_one_box([x1, y1, x2, y2], drawn_img,
                             color=cmaps[i % 2])

                # uncomment to draw skeletons on orig image
                PoseEstimator.draw_skeleton_from_keypts(
                    drawn_img, keypts, ignored_kp_idx=ig_kp_idx, color=cmaps[i % 2], thickness=crop_width // 150)

                # uncomment to draw keypts on orig image
                PoseEstimator.plot_keypts(
                    drawn_img, keypts, cmaps[i % 2], ignored_kp_idx=ig_kp_idx)

                # save heatmap plot
                PoseEstimator.plot_and_save_heatmap(
                    heatmap, f"{FLAGS.result_save_dir}/heatmap_{i}_{str(counter).zfill(6)}.jpg")

        if FLAGS.result_save_dir is not None:
            if FLAGS.inference_mode == "image":
                cv2.imwrite(
                    f"{FLAGS.result_save_dir}/frame_{str(counter).zfill(6)}.jpg", drawn_img)
            elif FLAGS.inference_mode == "video":
                vid_writer.write(drawn_img)
    counter += 1
    if FLAGS.debug:
        print(
            f"Time to process {counter} image(s)={time.time()-start_time:.3f}s")

    return final_result_list


def main():
    args = parse_arguments("Person Detection and Pose Estimation")
    run_pdet_pose(args.input_path,
                  model_name="ensemble_edet4_person_det_pose",
                  inference_mode=args.media_type,
                  det_threshold=args.detection_threshold,
                  save_result_dir=args.output_dir,
                  grpc_port=args.grpc_port,
                  debug=args.debug)


if __name__ == "__main__":
    main()
