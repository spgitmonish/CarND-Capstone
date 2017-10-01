# coding=utf-8
from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model


class TLClassifier(object):
    def __init__(self, model_uri=None):
        self.model = load_model('/home/dieslow/WORKSPACE/PROJECTS/CARZ/nanodegree/Capstone/ros/src/tl_detector/inceptv3beta0.h5')
        rospy.loginfo("THE MODEL HAS LOADED!")
        loc = "/home/dieslow/WORKSPACE/PROJECTS/CARZ/nanodegree/Capstone/ros/src/tl_detector/data/TrafficLightDataset/big_working_data/test/red/282579988.png"
        img = cv2.imread(loc)
        img = np.float32(img)
        img = preprocess_input(img)
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # seems weird here, but causes no problems
        img = np.expand_dims(img, 0)
        g_x = np.argmax(self.model.predict(img)[0])
        rospy.loginfo(np.argmax(g_x))

        self.predictionary = {
            0: TrafficLight.GREEN,
            1: TrafficLight.YELLOW,
            2: TrafficLight.RED,
            3: TrafficLight.UNKNOWN
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
​
        Args:
            image (cv::Mat): image containing the traffic light
​
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
​
        """
        rospy.loginfo("Classifyyyying!")

        img = np.float32(image)
        img = preprocess_input(img)
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # seems weird here, but causes no problems
        img = np.expand_dims(img, 0)
        g_x = np.argmax(self.model.predict(img)[0])

        rospy.loginfo("The label returned is %s", g_x)

        return self.predictionary[g_x]
