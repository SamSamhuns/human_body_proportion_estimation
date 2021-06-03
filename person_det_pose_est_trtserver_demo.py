from modules.triton_utils import parse_model_grpc, get_client_and_model_metadata_config
from modules.triton_utils import get_inference_responses, extract_data_from_media
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect, plot_one_box
from modules.pose_estimator import PoseEstimator

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
    boxes = results.as_numpy("ENSEMBLE_OUTPUT_FILTER_DET_BOXES")
    heatmaps = results.as_numpy("ENSEMBLE_OUTPUT_HEATMAPS")
    if len(heatmaps.shape) == 3:
        heatmaps = np.expand_dims(heatmaps, axis=0)
    return boxes.copy(), heatmaps


def run_demo_pdet_pose(media_filename,
                       model_name,
                       inference_mode='video',
                       det_threshold=0.55,
                       save_result_dir=None):  # set to None prevent saving
    FLAGS.media_filename = media_filename
    FLAGS.model_name = model_name
    FLAGS.inference_mode = inference_mode
    FLAGS.det_threshold = det_threshold
    FLAGS.result_save_dir = save_result_dir
    FLAGS.model_version = ""
    FLAGS.protocol = "grpc"
    FLAGS.url = '127.0.0.1:8994'
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.debug = True
    FLAGS.batch_size = 1
    FLAGS.fixed_input_width = 512
    FLAGS.fixed_input_height = 512
    # nose, reye, leye, rear, lear, rshoulder,
    # lshoulder, relbow, lelbow, rwrist, lwrist,
    # rhip, lhip, rknee, lknee, rankle, lankle
    # FLAGS.KEYPOINT_THRES_LIST = [0.65, 0.76, 0.73, 0.56, 0.44, 0.24, 0.13, 0.13, 0.34,
    #                              0.44, 0.38, 0.11, 0.13, 0.24, 0.15, 0.35, 0.30]
    FLAGS.KEYPOINT_THRES_LIST = [0.45, 0.46, 0.45, 0.40, 0.34, 0.10, 0.10, 0.10, 0.10,
                                 0.24, 0.30, 0.11, 0.10, 0.20, 0.10, 0.25, 0.20]
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
    w, h = None, None  # if set to None, no resizing of original input is done

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
        print("Image data was missing")
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
        boxes, heatmaps = postprocess(response, output_name)
        final_result_list.append([boxes, heatmaps])

        # display boxes on image array
        if FLAGS.result_save_dir is not None:
            drawn_img = all_reqested_images_orig[counter]
            drawn_img = resize_maintaining_aspect(drawn_img, w, h)

            # de-normalize bbox coords
            h, w, c = drawn_img.shape
            boxes *= [h, w, h, w]
            # for each human detected/crop/heatmap
            color_maps = [(255, 0, 255), (0, 255, 255)]
            for i, (heatmap, box) in enumerate(zip(heatmaps, boxes)):
                keypoints, keypoints_score = PoseEstimator.get_max_pred_keypoints_from_heatmap(
                    heatmap)
                x1, y1 = int(box[1]), int(box[0])  # top left
                x2, y2 = int(box[3]), int(box[2])  # bottom right

                # change coord axes of keypoints to match that of orig image
                _, hmap_height, hmap_width = heatmap.shape
                crop_width, crop_height = x2 - x1, y2 - y1
                # following is equi to this: x, y = int((x / hw) * cw) + x1, int((y / hh) * ch) + y1
                keypoints /= [hmap_width, hmap_height]
                keypoints *= [crop_width, crop_height]
                keypoints += [x1, y1]

                ignored_kp_idx = {i for i, score in enumerate(keypoints_score)
                                  if score < FLAGS.KEYPOINT_THRES_LIST[i]}

                # uncomment to plot bounding boxes
                plot_one_box([x1, y1, x2, y2], drawn_img,
                             color=color_maps[i % 2])

                # uncomment to draw skeletons on orig image
                PoseEstimator.draw_skeleton_from_keypoints(
                    drawn_img, keypoints, ignored_kp_idx=ignored_kp_idx, color=color_maps[i % 2])

                # uncomment to draw keypoints on orig image
                PoseEstimator.plot_keypoints(
                    drawn_img, keypoints, color_maps[i % 2], ignored_kp_idx=ignored_kp_idx)
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
    args = parse_arguments("Person Detection and Pose Estimation")
    run_demo_pdet_pose(args.input_path,
                       model_name="ensemble_person_det_and_pose",
                       inference_mode=args.media_type,
                       det_threshold=args.detection_threshold,
                       save_result_dir=args.output_dir)


if __name__ == "__main__":
    main()
