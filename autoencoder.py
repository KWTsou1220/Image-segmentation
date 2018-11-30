from sklearn.utils import shuffle

import tensorflow as tf
import pdb



def deconv_layer(x, filter_shape, output_shape, name, strides, padding):
    with tf.variable_scope(name):
        W = tf.get_variable(shape=filter_shape, 
                            name="weight", 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            dtype=tf.float32)
        b = tf.get_variable(shape=filter_shape[2], 
                            name="bias", 
                            initializer=tf.constant_initializer(0.1), 
                            dtype=tf.float32)
    return tf.add(tf.nn.conv2d_transpose(value=x, filter=W, output_shape=output_shape, strides=strides, padding=padding), b), W, b

def conv_layer(x, filter_shape, name, strides, padding):
    with tf.variable_scope(name):
        W = tf.get_variable(shape=filter_shape, 
                            name="weight", 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            dtype=tf.float32)
        b = tf.get_variable(shape=filter_shape[3], 
                            name="bias", 
                            initializer=tf.constant_initializer(0.1), 
                            dtype=tf.float32)
    return tf.add(tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding), b), W, b

def linear_layer(x, shape, name, params=None):
    # shape: a scalar indicating the number of neurons
    input_shape = x.shape[1]
    with tf.variable_scope(name):
        W = tf.get_variable(shape=[input_shape, shape], 
                            name="weight", 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            dtype=tf.float32)
        b = tf.get_variable(shape=shape, 
                            name="bias", 
                            initializer=tf.constant_initializer(0.1), 
                            dtype=tf.float32)
    return tf.add( tf.matmul(x, W), b ), W, b

class Autoencoder(object):
    
    def __init__(self, opt):
        # optimization setting
        self.LR = opt["LR"]
        self.input_shape  = opt["input_shape"]
        self.output_shape = opt["output_shape"]
        
        # neural network
        with tf.variable_scope("Model"):
            self.x = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name="input")
            self.t = tf.placeholder(dtype=tf.float32, shape=self.output_shape, name="target")
        self.neurons = [self.x,]
        self.weight  = []
        self.bias    = []
        self.batch_size = tf.shape(self.x)[0]
        
    def add_nonlinear(self, activation):
        with tf.variable_scope("Model"):
            y = activation(self.neurons[-1])
            self.neurons[-1] = y
    
    def add_nn(self, shape, name, non_linear=tf.nn.elu):
        with tf.variable_scope("Model"):
            if len(self.neurons[-1].shape)>2:
                _, height, width, ch = self.neurons[-1].shape.as_list()
                #pdb.set_trace()
                self.neurons[-1] = tf.reshape(self.neurons[-1], shape=[-1, height*width*ch])
            y, W, b = linear_layer(x=self.neurons[-1], shape=shape, name=name)
            self.weight.append(W)
            self.bias.append(b)
            self.neurons.append(non_linear(y))
        
    def add_rnn(self):
        pass
    def add_cnn(self, filter_shape, name, non_linear=tf.nn.elu, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope("Model"):
            y, W, b = conv_layer(self.neurons[-1], filter_shape, name, strides, padding)
            self.weight.append(W)
            self.bias.append(b)
            self.neurons.append(non_linear(y))
    
    def add_decnn(self, filter_shape, output_shape, name, strides=[1, 1, 1, 1], padding='SAME'):
        batch_size = self.neurons[-1].shape[0]
        output_shape = tf.stack([tf.shape(self.neurons[-1])[0], output_shape[1], output_shape[2], output_shape[3]])
        #output_shape = [batch_size, output_shape[1], output_shape[2], output_shape[3]]
        with tf.variable_scope("Model"):
            y, W, b = deconv_layer(self.neurons[-1], filter_shape, output_shape, name, strides, padding)
            self.weight.append(W)
            self.bias.append(b)
            self.neurons.append(y)
    
    def add_pooling(self, ksize, name, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope('Model'):
            y = tf.nn.max_pool(value=self.neurons[-1], ksize=ksize, strides=strides, padding=padding, name=name)
            self.neurons.append(y)
    
    def add_loss(self, loss):
        self.loss = loss(self.neurons[-1], self.t)
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss)
    
    def get_variable(self, name):
        with tf.variable_scope("Model/"+name, reuse=True):
            W = tf.get_variable(name="weight")
            b = tf.get_variabl32032e(name="bias")
            return W, b
        

    