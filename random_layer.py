from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot

class RandomLayerAbstract():
    def __init__(self,
                 hidden_neurons = None,
                 activation_function = None,
                 random_state = None):
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function
        self.random_state = random_state
        self.weights = None
        self.biases = None
        self.is_fitted = False

    def fit_transform(self, dataset):
        check_array(dataset)
        random_generator = check_random_state(self.random_state)
        self._generate_biases(dataset, random_generator)
        self._generate_weights(dataset, random_generator)
        self.is_fitted = True
        return self.transform(dataset)

    def transform(self, dataset):
        if self.is_fitted == False:
            raise ValueError('The random layer is not fitted')
        return self._compute_output_weights(dataset)

    def _generate_weights(self, dataset = None, random_generator = None):
        features = dataset.shape[1]
        self.weights = random_generator.normal(size = (features, self.hidden_neurons))
    
    def _generate_biases(self,  dataset = None, random_generator = None):
        self.biases = random_generator.normal(size = self.hidden_neurons)

    @abstractmethod
    def _compute_output_weights(self, dataset = None):
        pass
        

class RandomLayerGPU(RandomLayerAbstract):
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : tf.tanh(arg),
                 random_state = 0):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = 0)


    def _compute_output_weights(self, dataset = None):
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

        return output_weights

class RandomLayerCPU(RandomLayerAbstract):
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : np.tanh(arg),
                 random_state = 0):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = 0)


    def _compute_output_weights(self, dataset = None):
        return self.activation_function(safe_sparse_dot(dataset, self.weights) + self.biases) 