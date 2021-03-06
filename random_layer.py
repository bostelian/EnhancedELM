from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from utils.verbosity_manager import VerbosityManager
from utils.dataset_wrappers import flatten
from scipy.linalg import orth
from scipy.ndimage import convolve
from scipy.signal import convolve2d
#
import matplotlib.pyplot as plt
class RandomLayerAbstract():
    def __init__(self,
                 hidden_neurons = None,
                 activation_function = None,
                 random_state = None,
                 verbosity_mgr = False):
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function
        self.random_state = random_state
        self.verbosity_mgr = verbosity_mgr

    def fit_transform(self, dataset):
        self.verbosity_mgr.begin("fit_transform")
        self._fit(dataset)
        self.is_fitted = True
        self.verbosity_mgr.end()
        return self.transform(dataset)

    def transform(self, dataset):
        self.verbosity_mgr.begin("transform")
        if self.is_fitted == False:
            raise ValueError('The random layer is not fitted')
        hidden_activations = self._compute_hidden_activations(dataset)
        self.hidden_neurons = self._set_hidden_neurons(hidden_activations)
        self.verbosity_mgr.end()
        return hidden_activations

    def _set_hidden_neurons(self, output_weights = None):
        return output_weights.shape[1]

    @abstractmethod
    def _fit(self, dataset = None):
        pass

    @abstractmethod
    def _compute_hidden_activations(self, dataset = None):
        pass
        
class RandomLayerGeneral(RandomLayerAbstract):
    def __init__(self,
                hidden_neurons = 1000,
                activation_function = lambda arg : tf.tanh(arg),
                random_state = 0,
                verbosity_mgr = None):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = random_state,
                        verbosity_mgr = verbosity_mgr)

    def _fit(self, dataset = None):
            features = dataset.shape[1]
            self.weights = np.random.normal(size=(features, self.hidden_neurons)).astype("float32")
            self.biases = np.random.normal(size = self.hidden_neurons).astype('float32')


class RandomLayerGPU(RandomLayerGeneral):
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

    
    def _compute_hidden_activations(self, dataset = None):
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
            compute_hidden_activations = self.activation_function(dot_product_plc + biases_plc)

        with tf.Session() as sess:
            hidden_activations =  sess.run(compute_hidden_activations, feed_dict={dot_product_plc: dot_product, biases_plc : self.biases})
            sess.close()
        self.verbosity_mgr.end()
        return hidden_activations

class RandomLayerCPU(RandomLayerGeneral):
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
    
    def _compute_hidden_activations(self, dataset = None):
        self.verbosity_mgr.begin("compute_output_weights")
        hidden_activations = self.activation_function(safe_sparse_dot(dataset, self.weights) + self.biases)
        self.verbosity_mgr.end()
        return hidden_activations



class RandomLayerGeneralImage(RandomLayerAbstract):
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : np.tanh(arg),
                 random_state = 0,
                 LRF_threshold = 10,
                 border = 0,
                 data_shape = (0, 0),
                 verbosity_mgr = None):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = random_state,
                        verbosity_mgr = verbosity_mgr) 

    def _fit(self, dataset = None):
        self._set_colormaps(dataset)
        self._generate_weights(dataset)

    def _set_colormaps(self, dataset = None):
        self.colormaps = 1
        if len(dataset.shape) > 4 or len(dataset.shape) < 3:
            raise Exception("Invalid dataset shape")
        if len(dataset.shape) == 4:
            self.colormaps = dataset.shape[3]

    @abstractmethod
    def _generate_weights(self, dataset = None):
        pass

class RandomLayerLRF(RandomLayerGeneralImage):
    __NAME__ = "RandomLayerLRF"
    def __init__(self,
                 hidden_neurons = 1000,
                 activation_function = lambda arg : np.tanh(arg),
                 random_state = 0,
                 LRF_threshold = 10,
                 border = 0,
                 verbose = False):
        super().__init__(hidden_neurons = hidden_neurons,
                        activation_function = activation_function,
                        random_state = random_state,
                        verbosity_mgr = VerbosityManager(verbose = verbose, class_name = self.__NAME__))
        self.LRF_threshold = LRF_threshold
        self.border = border
    
    def _generate_weights(self, dataset = None):
        self.features = dataset.shape[1] * dataset.shape[2]
        self.weights = np.random.normal(size = (self.features, self.hidden_neurons)).astype('float32')
        LRF_mask = self._generate_LRF_mask(dataset)
        self.weights = np.multiply(LRF_mask, self.weights)

    def _generate_LRF_mask(self, dataset = None):
        self.verbosity_mgr.begin("generate_LRF_mask")
        LRF_mask = np.zeros((self.features, self.hidden_neurons), dtype="float32")
        indexMaxVal = dataset.shape[1] if dataset.shape[1] > dataset.shape[2] else dataset.shape[2]

        for neuron in range(0, self.hidden_neurons - 1):
            image_mask = np.zeros((dataset.shape[1], dataset.shape[2]))
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

    def _compute_hidden_activations(self, dataset = None):
        hidden_activations = np.zeros((dataset.shape[0], self.hidden_neurons))

        if self.colormaps > 1:
            for colormap in range(0, self.colormaps):           
                temp_dataset = dataset[:,:,:,colormap]
                temp_dataset = temp_dataset.reshape((dataset.shape[0], self.features))
                hidden_activations += self.activation_function(safe_sparse_dot(temp_dataset, self.weights))
        else:
            dataset = dataset.reshape(dataset.shape[0], self.features)
            hidden_activations = self.activation_function(safe_sparse_dot(dataset, self.weights))

        return hidden_activations


class RandomLayerConvolutional(RandomLayerGeneralImage):
    __NAME__ = "RandomLayerConvolutional"
    def __init__(self,
                 feature_maps = 10,
                 field_size = 4,
                 pooling_size = 3,
                 random_state = 0,
                 verbose = False):
        super().__init__(
                        random_state = random_state,
                        verbosity_mgr = VerbosityManager(verbose = verbose, class_name = self.__NAME__))
        self.feature_maps = feature_maps
        self.field_size = field_size
        self.pooling_size = pooling_size
    
    def _generate_weights(self, dataset = None):
        field_area = self.field_size * self.field_size
        self.weights = np.random.randint(low = -1, high = 1, size=(field_area, self.feature_maps))
        #self.weights = np.random.normal(size=(field_area, self.feature_maps))
        if field_area < self.feature_maps:
            self.weights = orth(self.weights.T).T
        else:
            self.weights = orth(self.weights)
 

    def _compute_hidden_activations(self, dataset = None):
        output_weights =[]
        for image in dataset:
            image_activation = np.array([], dtype=np.float32)
            for feature_map in range(0, self.feature_maps):
                convolved_data = self._pool(self._convolve(image, feature_map))
                image_activation = np.append(image_activation, convolved_data.flatten())
            output_weights.append(image_activation)
        return np.asarray(output_weights, dtype=np.float32)


    def _convolve(self, data = None, feature_map = None):
        result = np.zeros((data.shape[0], data.shape[1]))

        for colormap in range(0, self.colormaps):
            kernel = self.weights[:, feature_map].reshape(self.field_size, self.field_size)
            if self.colormaps > 1:
                result += convolve2d(data[:, :, colormap], kernel, mode='same', boundary = 'wrap') 
            else:
                result += convolve2d(data[:, :], kernel, mode='same', boundary = 'wrap') 
       
        return result

    def _pool(self, data = None):
        ones = np.ones((self.pooling_size, self.pooling_size))
        result = convolve2d(np.power(data[:,:], 2), ones, mode='same', boundary = 'wrap')
        return np.sqrt(result)
