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
import keras.backend.tensorflow_backend as K

CONFIDENCE_THRESHOLD = 0.8

class TLClassifier(object):
    """
    InceptionV3 model
    """
    def __init__(self):
        # Keras GPU optimization settings
        config = K.tf.ConfigProto()
        config.gpu_options.allow_growth=True
        K.set_session(K.tf.Session(config=config))

        # Load the model
        self.model = load_model(os.getcwd() + "/light_classification/" + "inceptv3beta0.h5")

        # Compile the model
        self.model._make_predict_function()
        self.predictionary = {
            0: TrafficLight.RED,
            1: TrafficLight.YELLOW,
            2: TrafficLight.GREEN,
            3: TrafficLight.UNKNOWN
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Image pre-processing pipleine
        img = np.float32(image)
        img = preprocess_input(img)
        img = cv2.resize(img, (299, 299))
        img = np.expand_dims(img, 0)
        # Execute model's predictions - return probability value for each of 4 classes
        probs = self.model.predict(img)[0]
        # get class with max probability
        g_x = np.argmax(probs)

        # reject if model is not confident about the prediction
        if probs[g_x] < CONFIDENCE_THRESHOLD:
            return TrafficLight.UNKNOWN

        # Swap label values as model was trained with different label values
        if g_x == 2:
            prediction = 0 # Red
        elif g_x == 0:
            prediction = 2 # Green
        elif g_x == 3:
            prediction = 1 # Yellow
        else:
            prediction = 3 # No light

        # Log the message
        rospy.loginfo("The label returned is %d", prediction)

        # Return the light state corresponding to the index
        return prediction

class TLClassifierSqueeze(object):
    """
    Squeezenet model
    """
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
    """
    VGG16 model
    """
    def __init__(self):
        # Load the model
        self.model = load_model(os.getcwd() + "/light_classification/" + "vgg16.h5")
        self.model._make_predict_function()
        self.predictionary = {
            0: TrafficLight.RED,
            1: TrafficLight.YELLOW,
            2: TrafficLight.GREEN,
            3: TrafficLight.UNKNOWN
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Image pre-processing pipeline
        img = cv2.resize(image, None, fx=0.5, fy=0.5)
        img = img.astype(np.float32)
        img = keras.applications.vgg16.preprocess_input(img)
        # Execute prediction
        probs = self.model.predict(np.array([img]), batch_size=1, verbose=1)[0]
        # get label with max probability
        g_x = np.argmax(probs)

        # reject if model is not confident
        if probs[g_x] < CONFIDENCE_THRESHOLD:
            return TrafficLight.UNKNOWN

        label = self.predictionary[g_x]
        rospy.loginfo("label: %d, conf: %f, %f, %f, %f", g_x, probs[0], probs[1], probs[2], probs[3])
        return label
