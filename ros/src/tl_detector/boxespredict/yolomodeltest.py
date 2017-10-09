#from darkflow.net.build import TFNet
import os
import cv2
import numpy as np
import json
from boxespredict.predict import *

# Class for testing the yolo model independent of the cythonized prediction files
class YOLOTest:
	def __init__(self, options):
		print('\nLoading from .pb and .meta')
		with tf.Session() as sess:
			with tf.gfile.FastGFile(options["pbLoad"], 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				graph = tf.import_graph_def(graph_def)
				self.inp = tf.get_default_graph().get_tensor_by_name('import/input:0')
				self.out = tf.get_default_graph().get_tensor_by_name('import/output:0')

				# Feed for testing
				self.feed = dict()

				# Metafile
				with open(options["metaLoad"], 'r') as fp:
					self.meta = json.load(fp)

				# Threshold for prediction filtering
				self.threshold = options["threshold"]
