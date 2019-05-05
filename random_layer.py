from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from utils.verbosity_manager import VerbosityManager

class RandomLayerAbstract():
    def __init__(self,
                 hidden_neurons = None,
                 activation_function = None,
                 random_state = None,
                 verbosity_mgr = False):
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function
        self.random_state = random_state
        self.weights = None
        self.biases = None
        self.is_fitted = False
        self.verbosity_mgr = verbosity_mgr

    def fit_transform(self, dataset):
        self.verbosity_mgr.begin("fit_transform")
        random_generator = check_random_state(self.random_state)
        self._generate_biases(dataset, random_generator)
        self._generate_weights(dataset, random_generator)
        self.is_fitted = True
        self.verbosity_mgr.end()
        return self.transform(dataset)

    def transform(self, dataset):
        self.verbosity_mgr.begin("transform")
        if self.is_fitted == False:
            raise ValueError('The random layer is not fitted')
        result = self._compute_output_weights(dataset)
        self.verbosity_mgr.end()
        return result

    def _generate_weights(self, dataset = None, random_generator = None):
        features = dataset.shape[1]
        self.weights = random_generator.normal(size = (features, self.hidden_neurons)).astype('float32')
    
    def _generate_biases(self,  dataset = None, random_generator = None):
        self.biases = random_generator.normal(size = self.hidden_neurons).astype('float32')

    @abstractmethod
    def _compute_output_weights(self, dataset = None):
        pass
        

class RandomLayerGPU(RandomLayerAbstract):
    __NAME__ = "RandomLayerGPU"
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : tf.tanh(arg),
                 random_state = 0,
                 verbose = False):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = random_state,
                        verbosity_mgr = VerbosityManager(verbose = verbose, class_name = self.__NAME__))


    def _compute_output_weights(self, dataset = None):
        self.verbosity_mgr.begin("compute_output_weights")
        dataset_plc = tf.placeholder(tf.float32, shape=dataset.shape)
        weights_plc = tf.placeholder(tf.float32, shape=self.weights.shape)
        with tf.device('/gpu:0'):
            compute_dot_product = tf.tensordot(dataset_plc, weights_plc, axes=1)

        with tf.Session() as sess:
            dot_product = sess.run(compute_dot_product, feed_dict={dataset_plc: dataset, weights_plc : self.weights})
            sess.close()

        dot_product_plc = tf.placeholder(tf.float32, shape=dot_product.shape)
        biases_plc = tf.placeholder(tf.float32, shape=self.biases.shape) 
        with tf.device('/gpu:0'):
            compute_output_weights = self.activation_function(dot_product_plc + biases_plc)

        with tf.Session() as sess:
            output_weights =  sess.run(compute_output_weights, feed_dict={dot_product_plc: dot_product, biases_plc : self.biases})
            sess.close()
        self.verbosity_mgr.end()
        return output_weights

class RandomLayerCPU(RandomLayerAbstract):
    __NAME__ = "RandomLayerCPU"
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : np.tanh(arg),
                 random_state = 0,
                 verbose = False):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = random_state,
                        verbosity_mgr = VerbosityManager(verbose = verbose, class_name = self.__NAME__))
    
    def _compute_output_weights(self, dataset = None):
        self.verbosity_mgr.begin("compute_output_weights")
        result = self.activation_function(safe_sparse_dot(dataset, self.weights) + self.biases)
        self.verbosity_mgr.end()
        return result

class RandomLayerLRF(RandomLayerAbstract):
    __NAME__ = "RandomLayerLRF"
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : np.tanh(arg),
                 random_state = 0,
                 LRF_threshold = 10,
                 border = 0,
                 data_shape = (0, 0),
                 verbose = False):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = random_state,
                        verbosity_mgr = VerbosityManager(verbose = verbose, class_name = self.__NAME__))
        self.LRF_threshold = LRF_threshold
        self.border = border
        self.data_shape = data_shape
    
    def _generate_weights(self, dataset = None, random_generator = None):
        features = dataset.shape[1]
        LRF_mask = self._generate_LRF_mask(features)
        self.weights = np.multiply(LRF_mask, random_generator.normal(size = (features, self.hidden_neurons)).astype('float32'))

    def _generate_LRF_mask(self, features):
        self.verbosity_mgr.begin("generate_LRF_mask")
        LRF_mask = np.zeros((features, self.hidden_neurons), dtype="float32")
        indexMaxVal = self.data_shape[0] if self.data_shape[0] > self.data_shape[1] else self.data_shape[1]
        for neuron in range(0, self.hidden_neurons - 1):
            image_mask = np.zeros(self.data_shape)
            index1 = np.zeros((2,1))
            index2 = np.zeros((2,1))
            while (index1[1]-index1[0]) * (index2[1] - index2[0]) < self.LRF_threshold:
                index1 =  self.border + np.sort(np.random.uniform(
                                                low = 0,
                                                high = indexMaxVal - 2 * self.border,
                                                size=(2,1)).astype("int"), axis=None)
                index2 =  self.border + np.sort(np.random.uniform(
                                                low = 0,
                                                high = indexMaxVal - 2 * self.border,
                                                size=(2,1)).astype("int"), axis=None)
            image_mask[index1[0]:index1[1]:1, index2[0]:index2[1]:1] = 1
            LRF_mask[:,neuron] = image_mask.flatten()
        self.verbosity_mgr.end()
        return LRF_mask

    def _compute_output_weights(self, dataset = None):
        return self.activation_function(safe_sparse_dot(dataset,self.weights))