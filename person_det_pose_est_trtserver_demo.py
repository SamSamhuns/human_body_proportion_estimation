from modules.triton_utils import parse_model_grpc, get_client_and_model_metadata_config
from modules.triton_utils import get_inference_responses, extract_data_from_media
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect, plot_one_box
from modules.pose_estimator import PoseEstimator

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
    boxes = results.as_numpy("ENSEMBLE_OUTPUT_FILTER_DET_BOXES")
    heatmaps = results.as_numpy("ENSEMBLE_OUTPUT_HEATMAPS")
    if len(heatmaps.shape) == 3:
        heatmaps = np.expand_dims(heatmaps, axis=0)
    return boxes.copy(), heatmaps


def run_pdet_pose(media_filename,
                  model_name="ensemble_person_det_and_pose",
                  # list with input heights, def avg male height
                  person_height=[175],
                  inference_mode='image',
                  det_threshold=0.70,
                  # set to None prevent saving
                  save_result_dir=None,
                  debug=True):
    FLAGS.media_filename = media_filename
    FLAGS.model_name = model_name
    FLAGS.p_height = person_height
    FLAGS.inference_mode = inference_mode
    FLAGS.det_threshold = det_threshold
    FLAGS.result_save_dir = save_result_dir
    FLAGS.model_version = ""
    FLAGS.protocol = "grpc"
    FLAGS.url = '127.0.0.1:8994'
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
    # FLAGS.KEYPOINT_THRES_LIST = [0.65, 0.76, 0.73, 0.56, 0.44, 0.24, 0.13, 0.13, 0.34,
    #                              0.44, 0.38, 0.11, 0.13, 0.24, 0.15, 0.35, 0.30]
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

    # all_reqested_images_orig will be [] if FLAGS.result_save_dir is None
    image_data, all_reqested_images_orig, fps = extract_data_from_media(
        FLAGS, preprocess, filenames, w, h)

    if len(image_data) == 0:
        print("Image data was missing")
        return -1

    trt_inf_data = (triton_client, input_name,
                    output_name, dtype, max_batch_size)
    # shift the coords of the bbox to sned more info to pose extractor
    x_shift = w // 17 if w is not None else image_data[0].shape[0] // 17
    y_shift = 0
    image_data_list = [image_data,
                       np.array([FLAGS.det_threshold], dtype=np.float32),
                       np.array([x_shift, y_shift], dtype=np.float32)]
    # get inference results
    responses = get_inference_responses(image_data_list, FLAGS, trt_inf_data)

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
            cmaps = [(255, 255, 0), (0, 0, 255)]
            for i, (heatmap, box) in enumerate(zip(heatmaps, boxes)):
                # save heatmap plot
                PoseEstimator.plot_and_save_heatmap(
                    heatmap, f"{FLAGS.result_save_dir}/heatmap_{i}_{str(counter).zfill(6)}.jpg")
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

                ig_kp_idx = {i for i, score in enumerate(keypoints_score)
                             if score < FLAGS.KEYPOINT_THRES_LIST[i]}

                # get estimations of body part lengths
                height_pixel = y2 - y1
                height_cm = FLAGS.p_height[min(i, len(FLAGS.p_height) - 1)]
                pixel_to_cm = height_cm / height_pixel
                dist_dict = PoseEstimator.get_keypoint_dist_dict(
                    pixel_to_cm, keypoints, ignored_kp_idx=ig_kp_idx)
                final_result_list[-1].append(dist_dict)

                # uncomment to plot bounding boxes
                plot_one_box([x1, y1, x2, y2], drawn_img,
                             color=cmaps[i % 2])

                # uncomment to draw skeletons on orig image
                PoseEstimator.draw_skeleton_from_keypoints(
                    drawn_img, keypoints, ignored_kp_idx=ig_kp_idx, color=cmaps[i % 2], thickness=crop_width // 150)

                # uncomment to draw keypoints on orig image
                PoseEstimator.plot_keypoints(
                    drawn_img, keypoints, cmaps[i % 2], ignored_kp_idx=ig_kp_idx)
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
                  model_name="ensemble_person_det_and_pose",
                  inference_mode=args.media_type,
                  det_threshold=args.detection_threshold,
                  save_result_dir=args.output_dir,
                  debug=args.debug)


if __name__ == "__main__":
    main()
