import cv2
import random
import argparse


class Flag_config:
    """stores configurations for prediction"""

    def __init__(self):
        pass


def parse_arguments(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--media_type',
                        default='image', type=str,
                        choices=('image', 'video'),
                        help='Type of Input: image, video')
    parser.add_argument('-i', '--input_path',
                        default='media/imgs/two people.jpg',  type=str,
                        help='Path to Input: Video File or Image file')
    parser.add_argument('-o', '--output_dir',
                        default='media/output',  type=str,
                        help='Output directory')
    parser.add_argument('-t', '--detection-threshold',
                        default=0.6,  type=float,
                        help='Detection Threshold')
    parser.add_argument('--debug',
                        default=True,
                        help='Debug Mode')

    return parser.parse_args()


def color_distance(rgb1, rgb2):
    """
    distance between two colors(3)
    """
    rm = 0.5 * (rgb1[0] + rgb2[0])
    d = sum((2 + rm, 4, 3 - rm) * (rgb1 - rgb2)**2)**0.5
    return d


def plot_one_box(bbox, img, wscale=1, hscale=1, color=None, label=None, line_thickness=None):
    """
    Plot one bounding box on image img
    args
        bboxes: bounding boxes in xyxy format (x1,y1,x2,y2)
        img: image in (H,W,C) numpy.ndarray fmt
        wscale: multiplication factor for width (default 1 if no scaling required)
        hscale: multiplication factor for height (default 1 if no scaling required)
    """
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1 = (int(bbox[0] * wscale), int(bbox[1] * hscale))
    c2 = (int(bbox[2] * wscale), int(bbox[3] * hscale))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def resize_maintaining_aspect(img, width, height):
    """
    If width and height are both None, no resize is done
    If either width or height is None, resize maintaining aspect
    """
    old_h, old_w, _ = img.shape

    if width is not None and height is not None:
        new_w, new_h = width, height
    elif width is None and height is not None:
        new_h = height
        new_w = (old_w * new_h) // old_h
    elif width is not None and height is None:
        new_w = width
        new_h = (new_w * old_h) // old_w
    else:
        # no resizing done if both width and height are None
        return img
    img = cv2.resize(img, (new_w, new_h))
    return img


def count_num_digits(num):
    """
    count number of digits in a given number
    """
    c = 0
    while num:
        num //= 10
        c += 1
    return c
