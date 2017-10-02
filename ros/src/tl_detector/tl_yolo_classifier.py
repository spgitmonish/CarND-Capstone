import rospy
import cv2
import numpy as np
import tensorflow as tf
import os

from darkflow.net.build import TFNet
from styx_msgs.msg import TrafficLight

CONFIDENCE_THRESHOLD = 0.8

class TLClassifier(object):
    def __init__(self):
	# The protocol buffer file and the .meta file
	# NOTE: The .met file is a JSON dump of everything necessary for post-processing such as anchors
	#       and labels
	# 30000 steps model
	#options = {"pbLoad": os.getcwd() + "/yolo_saved_graph/30000-tiny-yolo-voc-3c.pb", "metaLoad": os.getcwd() + "/yolo_saved_graph/30000-tiny-yolo-voc-3c.meta", "threshold": 0.1, "gpu": 1.0}

	# 40375 steps model
	options = {"pbLoad": os.getcwd() + "/yolo_saved_graph/40375-tiny-yolo-voc-3c.pb", "metaLoad": os.getcwd() + "/yolo_saved_graph/40375-tiny-yolo-voc-3c.meta", "threshold": 0.1, "gpu": 1.0}

	# Object of Darkflow
	self.tfnet = TFNet(options)

    def get_classification(self, image):
        """ Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
    	# Predict using the TFNet object
    	# NOTE: The results returned is in a format which has the following:
    	# 'result': [{'bottomright': { 'x': <value>, 'y': <value>},
    	#	          'confidence': <score_value>,
        # 	          'label': '<label_name>',
        # 	          'topleft': { 'x': <value>, 'y': <value>}]
        results = self.tfnet.return_predict(image)

        # By default the label to return is UNKNOWN
        label_to_return = TrafficLight.UNKNOWN
        # Value to store the predicted label
        predicted_label = None
        # Result to return is None too
        result_to_return = None
        # Variable to keep track of the prediction with the highest confidence
        highest_confidence = 0.0

        # Parse through the results and ensure the confidence score is > 0.8 for at least 1
        # the results, if that's the case then we need to return the bounding box result of the
        # one with the highest confidence
        if results != []:
            # Go through all the results
            for result in results:
                # Check if the threshold requirement is met
                if result['confidence'] > CONFIDENCE_THRESHOLD:
                    # Check if the prediction confidence is higher than the previous
                    if result['confidence'] > highest_confidence:
                        # Update the results with the highest confidence
                        highest_confidence = result['confidence']
                        # Store this result as the result to return
                        result_to_return = result
                        # Update the label
                        predicted_label = result['label']

        # Set the correct label for processing
        if predicted_label == "red_rect":
            label_to_return = TrafficLight.RED
        elif predicted_label == "orange_rect":
            label_to_return = TrafficLight.YELLOW
        elif predicted_label == "green_rect":
            label_to_return = TrafficLight.GREEN

        # Debug information
        if label_to_return != TrafficLight.UNKNOWN:
            rospy.loginfo("The light state is %d, with confidence %.3f", label_to_return, highest_confidence)

        return label_to_return, result_to_return
