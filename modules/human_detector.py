import tflite_runtime.interpreter as tflite
import cv2


class HumanDetectorTflite:
    """tf_lite model
    """

    def __init__(self, model_path="mnetv1_ssdlite_odet/ssd.tflite"):
        self.model = tflite.Interpreter(model_path=model_path)
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.model.allocate_tensors()
        self.inbatch, self.inheight, self.inwidth, self.inchannel = self.input_details[
            0]["shape"]

    def inference(self, frame, thres=0.59):
        """
        frame must be a numpy.ndarray of shape (H,W,C) in BGR space
        returns a tuple of bboxes and scores numpy.ndarrays
        """
        input_img = cv2.resize(cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB), (self.inwidth, self.inheight))
        self.model.set_tensor(self.input_details[0]['index'], [input_img])
        self.model.invoke()

        output_data_1 = self.model.get_tensor(self.output_details[0]["index"])
        output_data_2 = self.model.get_tensor(self.output_details[1]["index"])
        output_data_3 = self.model.get_tensor(self.output_details[2]["index"])

        bboxes = output_data_1[0]
        output_class_filter = (output_data_2 == 0.)[0]  # only use human class
        scores = output_data_3[0]

        bboxes = bboxes[output_class_filter]
        scores = scores[output_class_filter]
        bboxes = bboxes[scores >= thres]

        return bboxes, scores

    @staticmethod
    def get_people_crops(frame, bboxes):
        """
        args
            frame: numpy.ndarray of shape (H,W,C)
            bboxes: bounding boxes of shape [N,4] where each bbox is [y1,x1,y2,x2]
        return
            people_crops: array of crops imgs where each img has shape [H,W,C]
            lst_c1: bounding box coords for top left
            lst_c2: bounding box coords for bottom right
        """
        h, w, _ = frame.shape
        people_crops = []
        lst_c1 = []
        lst_c2 = []
        for i in range(bboxes.shape[0]):
            c1, c2 = ((int(max(bboxes[i, 1], 0) * w),
                       int(max(bboxes[i, 0], 0) * h)),
                      (int(max(bboxes[i, 3], 0) * w),
                       int(max(bboxes[i, 2], 0) * h)))
            people_crops.append(frame[c1[1]:c2[1], c1[0]:c2[0]])
            lst_c1.append(c1)
            lst_c2.append(c2)
        return people_crops, lst_c1, lst_c2
