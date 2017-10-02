import rospy
import cv2
import numpy as np

from styx_msgs.msg import TrafficLight
from squeezenet import *
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.models import model_from_yaml
from keras.applications.vgg16 import VGG16
import keras.applications.vgg16

class TLClassifier(object):
    def __init__(self):
        # Load the model
        self.model = load_model(os.getcwd() + "/light_classification/" + "inceptv3beta0.h5")
        # Compile the model
        self.model._make_predict_function()
        self.predictionary = {
            0: "RED",
            1: "ORANGE",
            2: "GREEN",
            3: "NOLIGHT"
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img = np.float32(image)
        img = preprocess_input(img)
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        g_x = np.argmax(self.model.predict(img)[0])
        # Kludge alert!
        g_x = (g_x - 2) % 4

        # Log the message
        rospy.loginfo("The label returned is %s", self.predictionary[g_x])

        # Return Unknown for now
        return TrafficLight.UNKNOWN

class TLClassifierSqueeze(object):
    def __init__(self):
        '''Hyperparameters'''
        self.num_classes = 3
        # Learning rate
        self.lr = 1e10
        # Number of epochs
        self.epochs = 2000
        # Batch Size
        self.batch_size = 16

        # Variable to store the session
        self.sess = tf.Session()

        # Get the necessary objects for testing
        self.model_logits, _, _, self.X, _, _ = createModel(self.num_classes, self.lr, self.epochs, self.batch_size)

        # Object of type Saver to save and load model
        model_saver = tf.train.Saver()

        # Restore the model from the checkpoint
        #model_saver.restore(self.sess, "squeezenet")
        model_saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd() + "/light_classification/"))

        # Initialize all the global variables
        self.sess.run(tf.global_variables_initializer())
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Run inference on image
        prediction = None
        prediction = inferOnImage(self.sess, self.model_logits, self.X, image)

        # Convert number into label just for debug
        prediction_label = None
        if prediction[0] == 0:
            prediction_label = "RED"
        elif prediction[0] == 1:
            prediction_label = "GREEN"
        elif prediction[0] == 2:
            prediction_label = "NOLIGHT"

        # Log the message
        rospy.loginfo("The label returned is %s", prediction_label)

        # Return Unknown for now
        return TrafficLight.UNKNOWN

class TLClassifierVGG16(object):
    def __init__(self):
        # Load the model
        self.model = load_model(os.getcwd() + "/light_classification/" + "vgg16.h5")

        self.model._make_predict_function()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img = cv2.resize(image, None, fx=0.5, fy=0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = keras.applications.vgg16.preprocess_input(img)
        pred = self.model.predict(np.array([img]), batch_size=1, verbose=1)
        g_x = np.argmax(pred[0])

        # Log the message
        rospy.loginfo("The label returned is %d", (g_x - 2) % 4)

        # Return Unknown for now
        return g_x