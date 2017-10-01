import rospy

from styx_msgs.msg import TrafficLight
from squeezenet import *

class TLClassifier(object):
    def __init__(self):
        '''Hyperparameters'''
        self.num_classes = 3
        # Learning rate
        self.lr = 1e10
        # Number of epochs
        self.epochs = 1000
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
        if prediction == 0:
            prediction_label = "RED"
        elif prediction == 1:
            prediction_label = "ORANGE"
        elif prediction == 2:
            prediction_label = "GREEN"

        # Log the message
        rospy.loginfo("The label returned is %s", prediction_label)

        # Return Unknown for now
        return TrafficLight.UNKNOWN
