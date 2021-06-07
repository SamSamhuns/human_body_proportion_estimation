import matplotlib.pyplot as plt
import onnxruntime
import numpy as np
import cv2

from typing import Dict


IDX_TO_KEYPOINTS = {
    0: "nose", 1: "reye", 2: "leye",
    3: "rear", 4: "lear",
    5: "rshoulder", 6: "lshoulder",
    7: "relbow", 8: "lelbow",
    9: "rwrist", 10: "lwrist",
    11: "rhip", 12: "lhip",
    13: "rknee", 14: "lknee",
    15: "rankle", 16: "lankle"}


class PoseEstimator:
    """onnx model
    """

    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.b, self.c, self.h, self.w = self.model.get_inputs()[0].shape

    @staticmethod
    def preprocess(frame_s, w=288, h=384) -> np.ndarray:
        """frame must be a numpy ndarray of shape (B,H,W,C) or (H,W,C)
        """
        if isinstance(frame_s, list):
            frame_s = np.array([cv2.resize(frame, (w, h))
                                for frame in frame_s])
        if len(frame_s) == 3:
            frame_s = np.expand_dims(frame_s, axis=0)

        def _preprocess(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (w, h)) / 255.0
            frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)
            return frame
        frames = [_preprocess(frame_s[i]) for i in range(frame_s.shape[0])]
        return np.array(frames)

    def inference(self, frame_s) -> np.ndarray:
        """
        args
            frame_s: numpy ndarray of shape (B,H,W,C) or (H,W,C)
        return
            heat_maps: ndarray of shape [batch, num_joints, map_height, map_width]
        """
        onnx_rt_inputs = {self.input_name: PoseEstimator.preprocess(
            frame_s, self.w, self.h)}
        onnx_rt_outs = self.model.run(None, onnx_rt_inputs)
        heat_maps = onnx_rt_outs[0]

        return heat_maps

    @staticmethod
    def plot_and_save_heatmap(heatmap, save_path) -> None:
        """
        args:
            heatmap: 3d numpy array of shape [channels, height, width]
            save_path: path where heatmap will be saved
        """
        # collapse all heatmaps into one 2d map
        heatmap_comb = np.sum(heatmap, axis=0)
        plt.figure(figsize=(20, 10))
        plt.imshow(heatmap_comb, cmap='hot', interpolation='nearest')
        plt.savefig(save_path)

    @staticmethod
    def get_max_pred_keypoints_from_heatmap(heatmap) -> tuple:
        """
        args
            heatmap: a numpy array of shape [num_joints, mheight, mwidth]
        return
            keypoints: coordinates of joints scaled to mheight, mwidth
            maxvals: confidence score of each joint
        """
        num_joints = heatmap.shape[0]
        width = heatmap.shape[2]
        # flatten pred shape for each joint
        heatmap_reshaped = heatmap.reshape((num_joints, -1))
        # get max score index and values
        maxidx, maxvals = np.argmax(
            heatmap_reshaped, 1), np.max(heatmap_reshaped, 1)
        maxidx, maxvals = maxidx.reshape(
            (num_joints, 1)), maxvals.reshape((num_joints, 1))

        preds = np.tile(maxidx, (1, 2)).astype(np.float32)
        preds[:, 0] = (preds[:, 0]) % width
        preds[:, 1] = np.floor((preds[:, 1]) / width)
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
        pred_mask = pred_mask.astype(np.float32)
        keypoints = preds * pred_mask
        return keypoints, maxvals

    @staticmethod
    def plot_keypoints(frame, keypoints, color, ignored_kp_idx=None) -> None:
        """
        args
            frame: numpy ndarray of shape (H,W,C)
            keypoints: array of keypoints of shape (N,2)
            color: color of keypoint (B,G,R)
            ignored_kp_idx: set of values in range [0, 16]
                indexes represented inside dict: IDX_TO_KEYPOINTS
        """
        ignored_kp_idx_set = set(
            ignored_kp_idx) if ignored_kp_idx is not None else {}
        for i, (x, y) in enumerate(keypoints):
            if i not in ignored_kp_idx_set:
                cv2.putText(frame,
                            f"{i}",
                            (x, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=color)
                cv2.circle(frame,
                           (x, y),
                           frame.shape[0] // 150,
                           color,
                           thickness=-1)

    @staticmethod
    def _get_kp_dict(keypoints, ignored_kp_idx):
        """ Get full set of kp list and use set
        """
        ignored_kp_idx = set(
            ignored_kp_idx) if ignored_kp_idx is not None else {}
        # set to use for skeleton connection, use_set
        uset = {val for key, val in IDX_TO_KEYPOINTS.items()
                if key not in ignored_kp_idx}

        # from the screen viewer not the images perspective
        (nose, reye, leye, rear, lear, rshoulder, lshoulder,
         relbow, lelbow, rwrist, lwrist, rhip, lhip,
         rknee, lknee, rankle, lankle) = keypoints

        # get chest and crotch
        if {'rshoulder', 'lshoulder'} <= uset:
            chest = [int(rshoulder[0] + lshoulder[0]) // 2,
                     int(rshoulder[1] + lshoulder[1]) // 2]
            uset.add('chest')
        if {"rhip", "lhip"} <= uset:
            crotch = [int(rhip[0] + lhip[0]) // 2,
                      int(rhip[1] + lhip[1]) // 2]
            uset.add('crotch')

        # reverse perspective to that of the image person
        kp_dict = {"shoulder": [{"rshoulder", "lshoulder"}, (rshoulder, lshoulder)],
                   "torso": [{"crotch", "chest"}, (crotch, chest)],
                   "lshoulder_lelbow": [{"rshoulder", "relbow"}, (rshoulder, relbow)],
                   "rshoulder_relbow": [{"lshoulder", "lelbow"}, (lshoulder, lelbow)],
                   "lwrist_lelbow": [{"rwrist", "relbow"}, (rwrist, relbow)],
                   "rwrist_relbow": [{"lwrist", "lelbow"}, (lwrist, lelbow)],
                   "rhip_lhip": [{"lhip", "rhip"}, (lhip, rhip)],
                   "rhip_rknee": [{"lhip", "lknee"}, (lhip, lknee)],
                   "lhip_lknee": [{"rhip", "rknee"}, (rhip, rknee)],
                   "rankle_rknee": [{"lankle", "lknee"}, (lankle, lknee)],
                   "lankle_lknee": [{"rankle", "rknee"}, (rankle, rknee)]}

        # filter points that are ignored and set them to None
        name_kp_dict = {k: v[1] if v[0] <= uset else None
                        for k, v in kp_dict.items()}
        return name_kp_dict

    @staticmethod
    def _calc_dist_betw_keypoints(kp_dict: Dict[str, tuple]) -> None:
        for part_name, part_kp in kp_dict.items():
            if part_kp is not None:  # if required points are not missing
                p1, p2 = map(np.asarray, part_kp)
                kp_dict[part_name] = np.linalg.norm(p1 - p2)

    @staticmethod
    def _draw_line_bet_keypoints(kp_dict: Dict[str, tuple], frame: np.ndarray,
                                 color: tuple, thickness: int) -> Dict[str, float]:
        for part_name, part_kp in kp_dict.items():
            if part_kp is not None:  # if required points are not missing
                p1, p2 = map(tuple, part_kp)
                cv2.line(frame, p1, p2, color, thickness)

    @staticmethod
    def get_keypoint_dist_dict(pixel_to_cm, keypoints, ignored_kp_idx=None) -> Dict[str, float]:
        # get connec name, end point name + end point coord dict
        name_kp_dict = PoseEstimator._get_kp_dict(keypoints, ignored_kp_idx)
        # replace dict point names with their dist
        PoseEstimator._calc_dist_betw_keypoints(name_kp_dict)
        # conv pixels values to cm
        dist_dict = {k: v * pixel_to_cm for k, v in name_kp_dict.items()}
        return dist_dict

    @staticmethod
    def draw_skeleton_from_keypoints(frame, keypoints, ignored_kp_idx=None, color=(0, 0, 255), thickness=1) -> None:
        """
        args:
            frame: numpy ndarray of shape (H,W,C)
            keypoints: array of keypoints of shape (N,2)
            ignored_kp_idx: set containing numerical indexes to ignore [0, 16]
        """
        # get connec name, end point name + end point coord dict
        name_kp_dict = PoseEstimator._get_kp_dict(keypoints, ignored_kp_idx)
        # plot lines between keypoints that are valid
        PoseEstimator._draw_line_bet_keypoints(name_kp_dict, frame, color, thickness)
