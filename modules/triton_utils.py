import os
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from modules.utils import count_num_digits


def get_client_and_model_metadata_config(FLAGS):
    try:
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        return -1

    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        return -1

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        return -1

    return triton_client, model_metadata, model_config


def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    # Set the input data
    inputs = []
    inputs.append(
        grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = []
    for i in range(len(output_name)):
        outputs.append(grpcclient.InferRequestedOutput(
            output_name[i], class_count=FLAGS.classes))

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def parse_model_grpc(model_metadata, model_config):
    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata_name = []
    for i in range(len(model_metadata.outputs)):
        output_metadata_name.append(model_metadata.outputs[i].name)
    s1 = input_metadata.shape[1]
    s2 = input_metadata.shape[2]
    s3 = input_metadata.shape[3]
    return (model_config.max_batch_size, input_metadata.name,
            output_metadata_name, s1, s2, s3, input_config.format,
            input_metadata.datatype)


def extract_data_from_media(FLAGS, preprocess_func, media_filenames, w, h):
    image_data = []
    all_reqested_images_orig = []
    leading_zeros, fmt, fps = None, None, None

    for filename in media_filenames:
        if FLAGS.inference_mode == "image":
            try:
                # if an image path is provided instead of a numpy H,W,C image
                if isinstance(filename, str) and os.path.isfile(filename):
                    filename = cv2.imread(filename)
                image_data.append(preprocess_func(filename, w, h))
                if FLAGS.frames_save_dir is not None:
                    all_reqested_images_orig.append(filename)

                # dummy values
                leading_zeros = count_num_digits(1)
                fmt = f"0{leading_zeros}d"
                fps = 1
            except Exception as e:
                print(f"{e}. Failed to process image {filename}")
        elif FLAGS.inference_mode == "video":
            try:
                cap = cv2.VideoCapture(filename)
                vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) - 10  # reduce orig fps by 10
                if vid_length > 10000:
                    raise Exception("Video must have less than 10000 frames")

                # for saving frames of videos in correct format
                leading_zeros = count_num_digits(vid_length + 1)
                fmt = f"0{leading_zeros}d"

                # check num of channels
                ret, frame = cap.read()
                if ret and frame.shape[-1] != 3:
                    raise Exception("Video must have 3 channels")

                # set opencv reader to vid start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                orig_vid, vid = [], []
                i = 0
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        if FLAGS.frames_save_dir is not None:
                            orig_vid.append(np.copy(frame))
                        vid.append(preprocess_func(frame, w, h))
                    else:
                        break
                    i += 1
                image_data = vid
                all_reqested_images_orig = orig_vid
                cap.release()
            except Exception as e:
                print(f"{e}. Failed to process video {filename}")
    return image_data, all_reqested_images_orig, fps, fmt, leading_zeros
