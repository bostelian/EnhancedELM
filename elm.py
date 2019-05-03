from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import scipy.linalg

from random_layer import RandomLayerCPU, RandomLayerGPU
from utils.stopwatch import Stopwatch
from numba import jit


class ELMAbstract(ABC):
        def __init__(self,
                hidden_layer = None,
                classifier = None,
                C = None,
                binarizer = None,
                stopwatch = None
                ):
            self.hidden_layer = hidden_layer
            self.C = C
            self.binarizer = binarizer
            self.stopwatch = stopwatch
            self.output_weights = None
            self.is_fitted = False
            self.running_times = {}
            self.classifier = classifier
        
        def fit(self, dataset = None, labels = None):
            self.stopwatch.start()
            hidden_activations = self.hidden_layer.fit_transform(dataset)
            if self.classifier == None:
                labels_bin = self._binarize(labels)
                self._fit_classifier(labels_bin, hidden_activations)
            else:
                self.classifier.fit(hidden_activations, labels)
            self.is_fitted = True
            self.running_times['fit'] = self.stopwatch.stop()
            self.stopwatch.clear()
            return self
      
        def predict(self, dataset = None):
            if self.is_fitted == False:
                raise Exception("Classifier is not fitted")
            self.stopwatch.start()
            hidden_activations = self.hidden_layer.transform(dataset)
            if self.classifier == None:
                predictions = self._unbinarize(self._predict_classifier(hidden_activations))
            else:
                predictions = self.classifier.predict(hidden_activations)
            self.running_times['predict'] = self.stopwatch.stop()
            self.stopwatch.clear()
            return predictions
    
        def _binarize(self, labels = None):
            return np.float32(self.binarizer.fit_transform(labels))

        def _unbinarize(self, labels_bin = None):
            return self.binarizer.inverse_transform(labels_bin)
        
        def get_running_times(self):
            return self.running_times

        @abstractmethod
        def _fit_classifier(self, targets = None, hidden_activations = None):
            pass

        @abstractmethod
        def _predict_classifier(self, hidden_activations = None):
            pass

class ELMCPU(ELMAbstract):
    def __init__(self,
                hidden_layer = RandomLayerCPU(),
                classifier = None,
                C = 1.0,
                binarizer = LabelBinarizer(-1, 1),
                stopwatch = Stopwatch()):
        super().__init__(hidden_layer = hidden_layer,
                            classifier=classifier,
                            C = C,
                            binarizer = binarizer,
                            stopwatch = stopwatch)


    def _fit_classifier(self, targets = None, hidden_activations = None):
        hidden_neurons = self.hidden_layer.hidden_neurons
        self.output_weights =  scipy.linalg.solve(
                                np.eye(hidden_neurons) / self.C + np.dot(hidden_activations.T,hidden_activations),
                                np.dot(hidden_activations.T,targets))


    def _predict_classifier(self, hidden_activations = None):
        return np.dot(hidden_activations, self.output_weights)

class ELMGPU(ELMAbstract):
    def __init__(self,
                hidden_layer = RandomLayerGPU(),
                classifier = None,
                C = 1.0,
                binarizer = LabelBinarizer(),
                stopwatch = Stopwatch()):
        super().__init__(hidden_layer = hidden_layer,
                            classifier=classifier,
                            C = C,
                            binarizer = binarizer,
                            stopwatch = stopwatch)
    
    def _fit_classifier(self, targets = None, hidden_activations = None):
        hidden_neurons = self.hidden_layer.hidden_neurons
        identity = np.eye(hidden_neurons)

        hidden_activations_plc = tf.placeholder(tf.float32, shape=hidden_activations.shape)
        hidden_activations_t_plc = tf.placeholder(tf.float32, shape=hidden_activations.T.shape)
        targets_plc = tf.placeholder(tf.float32, shape=targets.shape)
        identity_plc = tf.placeholder(tf.float32, shape=identity.shape)

        with tf.device('/gpu:0'):
            compute_dot1 = tf.tensordot(hidden_activations_t_plc, hidden_activations_plc, axes=1)

        with tf.Session() as sess:
            dot1 =  sess.run(compute_dot1, feed_dict={hidden_activations_t_plc: hidden_activations.T,
                                                 hidden_activations_plc : hidden_activations})
            sess.close()
        
        dot1_plc = tf.placeholder(tf.float32, shape=dot1.shape)

        with tf.device('/gpu:0'):
            compute_dot2 = tf.tensordot(hidden_activations_t_plc, targets_plc, axes=1)

        with tf.Session() as sess:
            dot2 =  sess.run(compute_dot2, feed_dict={hidden_activations_t_plc: hidden_activations.T,
                                                 targets_plc : targets})
            sess.close()
        
        dot2_plc = tf.placeholder(tf.float32, shape=dot2.shape)

        with tf.device('/gpu:0'):
            compute_inverse = tf.matrix_solve(identity_plc  + dot1_plc,
                                       dot2_plc)
        with tf.Session() as sess:                            
            output_weights = sess.run(compute_inverse, feed_dict={identity_plc: identity, dot1_plc : dot1, dot2_plc : dot2})
            sess.close()

        return output_weights

    def _predict_classifier(self, hidden_activations = None):
        hidden_activations_plc = tf.placeholder(tf.float32, shape=hidden_activations.shape)
        output_weights_plc = tf.placeholder(tf.float32, shape=self.output_weights.shape)

        with tf.device('/gpu:0'):
            compute_dot = tf.tensordot(hidden_activations_plc, output_weights_plc, axes=1)

        with tf.Session() as sess:
            predictions =  sess.run(compute_dot, feed_dict={hidden_activations_plc: hidden_activations,
                                                 output_weights_plc : self.output_weights})
            sess.close()    
        
        return predictions



