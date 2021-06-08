from modules.triton_utils import parse_model_grpc, get_client_and_model_metadata_config
from modules.triton_utils import get_inference_responses, extract_data_from_media
from modules.utils import Flag_config, parse_arguments, resize_maintaining_aspect
from modules.pose_estimator import PoseEstimator

import numpy as np
import time
import cv2
import os
FLAGS = Flag_config()


def preprocess(img, width=288, height=384, new_type=np.float32):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = resize_maintaining_aspect(img, width, height).astype(new_type)
    img = np.transpose(img, (2, 0, 1))
    return img


def postprocess(results, output_name):
    output_set = set(output_name)
    if "output" in output_set:
        heatmaps = results.as_numpy("output")    # batched hrnet
    elif "output_2" in output_set:
        heatmaps = results.as_numpy("output_2")  # higherhrnet
    return heatmaps


def run_demo_pose_est(media_filename,
                      model_name,
                      person_height=[175],
                      inference_mode="video",
                      det_threshold=0.55,
                      save_result_dir=None,  # set to None prevent saving
                      debug=True):
    FLAGS.det_threshold = det_threshold
    FLAGS.media_filename = media_filename
    FLAGS.model_name = model_name
    FLAGS.person_height = person_height
    FLAGS.inference_mode = inference_mode
    FLAGS.result_save_dir = save_result_dir
    FLAGS.model_version = ""
    FLAGS.protocol = "grpc"
    FLAGS.url = '127.0.0.1:8994'
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.debug = debug
    FLAGS.batch_size = 1
    FLAGS.fixed_input_width = 512
    FLAGS.fixed_input_height = 512
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

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model_grpc(
        model_metadata, model_config.config)

    # check for dynamic input shapes
    if h == -1:
        h = FLAGS.fixed_input_height
    if w == -1:
        w = FLAGS.fixed_input_width

    filenames = []
    if os.path.isdir(FLAGS.media_filename):
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
    responses = get_inference_responses([image_data], FLAGS, trt_inf_data)

    if FLAGS.inference_mode == "video" and FLAGS.result_save_dir is not None:
        vid_writer = cv2.VideoWriter(f"{FLAGS.result_save_dir}/res_video.mp4",
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    counter = 0
    for response in responses:
        heatmaps = postprocess(response, output_name)
        # for batch_sz of 1, there will only one heatmap
        for heatmap in heatmaps:
            # save heatmap plot
            PoseEstimator.plot_and_save_heatmap(
                heatmap, f"{FLAGS.result_save_dir}/heatmap_{str(counter).zfill(6)}.jpg")
            keypts, conf = PoseEstimator.get_max_pred_keypts_from_heatmap(
                heatmap)

            # display boxes on image array
            if FLAGS.result_save_dir is not None:
                drawn_img = all_reqested_images_orig[counter]
                # drawn_img = resize_maintaining_aspect(drawn_img, w, h)

                _, hmap_height, hmap_width = heatmap.shape
                img_height, img_width, _ = drawn_img.shape
                keypts /= [hmap_width, hmap_height]
                keypts *= [img_width, img_height]

                # uncomment to draw skeletons on orig image
                PoseEstimator.draw_skeleton_from_keypts(
                    drawn_img, keypts, ignored_kp_idx=None, color=(0, 0, 255), thickness=2)

                PoseEstimator.plot_keypts(
                    drawn_img, keypts, (0, 0, 255))
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


def main():
    args = parse_arguments("Single Person Pose Estimation")
    run_demo_pose_est(args.input_path,
                      model_name="higherhrnet",
                      inference_mode=args.media_type,
                      det_threshold=args.detection_threshold,
                      save_result_dir=args.output_dir,
                      debug=args.debug)


if __name__ == "__main__":
    main()
