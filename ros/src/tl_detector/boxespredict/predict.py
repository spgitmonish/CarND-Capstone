import os
import time
import numpy as np
import tensorflow as tf
import pickle
import cv2
from boxespredict.boxes import *

def findboxes(net_out, meta):
	#  file
	meta = meta
	boxes = list()
	boxes = box_constructor(meta, net_out)
	return boxes

def resize_input(im, meta):
	h, w, c = meta['inp_size']
	imsz = cv2.resize(im, (w, h))
	imsz = imsz / 255.
	imsz = imsz[:,:,::-1]
	return imsz

def process_box(b, h, w, threshold, meta):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = meta['labels'][max_indx]
	if max_prob > threshold:
		left  = int ((b.x - b.w/2.) * w)
		right = int ((b.x + b.w/2.) * w)
		top   = int ((b.y - b.h/2.) * h)
		bot   = int ((b.y + b.h/2.) * h)
		if left  < 0    :  left = 0
		if right > w - 1: right = w - 1
		if top   < 0    :   top = 0
		if bot   > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)

	return None

def return_predict(im, yolo_test):
    assert isinstance(im, np.ndarray), 'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = resize_input(im, yolo_test.meta)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {yolo_test.inp : this_inp}

    with tf.Session() as sess:
        out = sess.run(yolo_test.out, feed_dict)[0]
        boxes = findboxes(out, yolo_test.meta)
        threshold = yolo_test.threshold
        boxesInfo = list()

        for box in boxes:
            tmpBox = process_box(box, h, w, threshold, yolo_test.meta)
            if tmpBox is None:
                continue
            boxesInfo.append({
                "label": tmpBox[4],
                "confidence": tmpBox[6],
                "topleft": {
                    "x": tmpBox[0],
                    "y": tmpBox[2]},
                "bottomright": {
                    "x": tmpBox[1],
                    "y": tmpBox[3]}
            })
        return boxesInfo
