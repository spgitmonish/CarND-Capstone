import numpy as np

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

# Calculates the overlap between the ground truth and prediction
def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 /2.
    l2 = x2 - w2 /2.
    left = max(l1, l2)
    r1 = x1 + w1 /2.
    r2 = x2 + w2 /2.
    right = min(r1, r2)
    return right - left;

# Calculates the interection between ground truth and prediction
def box_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    w = overlap(ax, aw, bx, bw)
    h = overlap(ay, ah, by, bh)
    if w < 0 or h < 0:
        return 0
    area = w * h

    return area

# Calculates the union between ground truth and prediction
def box_union(ax, ay, aw, ah, bx, by, bw, bh):
    intersection = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    union = aw * ah + bw * bh - intersection

    return union

# Calculates the IOU between ground truth and prediction
def box_iou(ax, ay, aw, ah, bx, by, bw, bh):
    return box_intersection(ax, ay, aw, ah, bx, by, bw, bh) / box_union(ax, ay, aw, ah, bx, by, bw, bh)
