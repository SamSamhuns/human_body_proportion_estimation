import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MyModule(tf.Module):
    def __init__(self, model, **kwargs):
        super(MyModule, self).__init__(**kwargs)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.uint8),
                                  tf.TensorSpec(shape=(1), dtype=tf.float32),
                                  tf.TensorSpec(shape=(2), dtype=tf.float32)])
    def update_signature(self, edet_input_image, det_thres, det_xy_change):  # edet_input_image is the input name
        output = self.model(edet_input_image)
        det_boxes = output[0][0]    # 100,4
        det_scores = output[1][0]   # 100
        det_classes = output[2][0]  # 100
        fil_det_boxes = tf.zeros([100, 4], dtype=tf.float32)  # placeholder

        # postprocessing which is preprocessing for the next model in pipeline done here
        human_class_idx = tf.squeeze(tf.where(det_classes == 1.0), axis=1)

        # det_thres = 0.65
        # filter non human classes from det_scores and det boxes
        fil_det_scores = tf.gather(
            det_scores, human_class_idx, axis=0)  # (None,)
        fil_det_boxes = tf.gather(
            det_boxes, human_class_idx, axis=0)   # (None, 4,), y1, x1, y2, x2

        # filter low score human detections from bboxes, only take 3 boxes at max
        high_scores_idx = tf.squeeze(
            tf.where(fil_det_scores >= det_thres), axis=1)  # (None,)
        fil_det_boxes = tf.gather(
            fil_det_boxes, high_scores_idx, axis=0)[:3]     # (None, 4,)

        h = tf.shape(edet_input_image)[1]
        w = tf.shape(edet_input_image)[2]
        x_change = det_xy_change[0]
        y_change = det_xy_change[1]
        # increase the size of bounding boxes
        hf = tf.cast(h, dtype=tf.float32)
        wf = tf.cast(w, dtype=tf.float32)
        y1 = fil_det_boxes[:, 0] - y_change  # (3, 1)
        y1 = tf.clip_by_value(y1, clip_value_min=0, clip_value_max=hf)
        x1 = fil_det_boxes[:, 1] - x_change  # (3, 1)
        x1 = tf.clip_by_value(x1, clip_value_min=0, clip_value_max=wf)
        y2 = fil_det_boxes[:, 2] + y_change  # (3, 1)
        y2 = tf.clip_by_value(y2, clip_value_min=0, clip_value_max=hf)
        x2 = fil_det_boxes[:, 3] + x_change  # (3, 1)
        x2 = tf.clip_by_value(x2, clip_value_min=0, clip_value_max=wf)
        fil_det_boxes_expand = tf.stack([y1, x1, y2, x2], axis=1)

        # get a [h,w,h,w] tensor where h,w = orig image height,width
        edet_input_image_tensor = tf.cast([h, w, h, w], dtype=tf.float32)
        # normalize bbox coords to [0,1] by dividing by the orig image dimensions
        fil_det_boxes = fil_det_boxes_expand / edet_input_image_tensor

        edet_input_image = tf.cast(edet_input_image, dtype=tf.float32)
        edet_input_image /= 255.0  # normalize to range 0 to 1
        crop_size = (384, 288)
        batch_size = 1  # since we will be streaming images to this human det model
        box_indices = tf.random.uniform(shape=(len(fil_det_boxes),),
                                        minval=0,
                                        maxval=batch_size,
                                        dtype=tf.int32)
        human_crops = tf.image.crop_and_resize(edet_input_image,
                                               fil_det_boxes,
                                               box_indices,
                                               crop_size)

        def _human_det():
            return human_crops

        def _no_human_det():
            return tf.zeros([1, 384, 288, 3], dtype=tf.dtypes.float32)

        human_crops = tf.cond(
            tf.equal(tf.shape(human_crops)[0], 0), _no_human_det, _human_det)
        human_crops = tf.transpose(human_crops, perm=[0, 3, 1, 2])

        return {"detection_boxes": det_boxes,
                "detection_scores": det_scores,
                "detection_classes": det_classes,
                "filtered_boxes": fil_det_boxes,
                "human_crops": human_crops}


def main():
    model = tf.saved_model.load(
        "../extra_models/edetlite4/1/model.savedmodel/",
        tags="serve")

    module = MyModule(model)
    save_path = "edetlite4_modified_new/1/model.savedmodel"
    os.makedirs(save_path, exist_ok=True)

    tf.saved_model.save(module,
                        save_path,
                        signatures={"serving_default": module.update_signature})


if __name__ == "__main__":
    main()
