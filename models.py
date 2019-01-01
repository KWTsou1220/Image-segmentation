from layer import conv_layer, pooling_layer, deconv_layer
import tensorflow as tf
import pdb

def sparse_sigmoid(x, is_train, p=38, r=20):
    # x: (data_size, height, width, 1)
    with tf.variable_scope('thresholding'):
        batch_size, height, width, ch = x.shape.as_list()
        x = tf.reshape(x, [-1, height*width*ch])
        batch_t = tf.nn.top_k(tmp, k=p)
        batch_t = tf.reduce_mean(batch_t[:, -1])
        
        avg_t = tf.get_variable(name='threshold', 
                                shape=[1], 
                                initializer=tf.constant_initializer(0.0), 
                                trainable=False)
        if is_train:
            avg_t_assign_op = tf.assign(avg_t, 0.9*avg_t + 0.1*batch_t)
            with tf.control_dependencids([avg_t_assign_op]):
                return tf.nn.sigmoid(r*(x - batch_t))
        else:
            return tf.nn.sigmoid(r*(x-avg_t))

class SparseAE(object):
    def __init__(self, LR, input_shape, output_shape, model_name='Sparse-AutoEncoder'):
        # optimization setting
        self.LR = LR
        
        # naming setting
        self.model_name = model_name
        
        # model setting
        self.input_shape = input_shape
        self.output_shape = output_shape
        with tf.variable_scope(self.model_name):
            self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
            self.x = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t = tf.placeholder(dtype=tf.float32, shape=self.output_shape, name='output')
            with tf.variable_scope('Encoder'):
                self.feature_c, self.detect_c = self.Encode(self.x)
                self.c = tf.multiply(self.feature_c, self.detect_c)
            with tf.variable_scope('Decoder'):
                self.y = self.Decode(self.c)
    
    def Encode(self, x):
        h = conv_layer(x, filter_shape=[5, 5, 1, 64], strides=[1, 3, 3, 1], name='L1') # (90, 300)
        h = conv_layer(h, filter_shape=[5, 5, 64, 64], name='L2') # (90, 300)
        h = conv_layer(h, filter_shape=[5, 5, 64, 128], strides=[1, 2, 2, 1], name='L3') # (45, 150)
        h = conv_layer(h, filter_shape=[5, 5, 128, 128], name='L4') # (45, 150)
        h = conv_layer(h, filter_shape=[5, 5, 128, 256], strides=[1, 3, 3, 1], name='L5') # (15, 50)
        feature_h = conv_layer(h, filter_shape=[5, 5, 256, 256], name='L6', non_linear=tf.nn.relu) # (15, 50)
        detect_h = conv_layer(h, filter_shape=[5, 5, 256, 1], name='L6', non_linear=None) # (15, 50)
        detect_h = sparse_sigmoid(detect_h, is_train=self.is_train)
        return feature_h, detect_h
    def Decode(self, c):
        h = conv_layer(c, filter_shape=[5, 5, 256, 256], name='L1') # (15, 50)
        h = deconv_layer(h, filter_shape=[5, 5, 128, 256], strides=[1, 3, 3, 1], output_shape=[-1, 45, 150, 128], name='L2') # (45, 150)
        h = conv_layer(h, filter_shape=[5, 5, 128, 128], name='L3') # (45, 150)
        h = deconv_layer(h, filter_shape=[5, 5, 64, 128], strides=[1, 2, 2, 1], output_shape=[-1, 90, 300, 64], name='L4') # (90, 300)
        h = conv_layer(h, filter_shape=[5, 5, 64, 64], name='L5') # (90, 300)
        h = deconv_layer(h, filter_shape=[5, 5, 1, 64], strides=[1, 3, 3, 1], output_shape=[-1, 270, 900, 1], name='L6') # (270, 900)
        return h
        
    def optimize(self, loss):
        self.loss = loss(self.y, self.t)
        self.training = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
    
class UNet(object):
    def __init__(self, LR, input_shape, output_shape, model_name='U-Net'):
        # optimization setting
        self.LR = LR
        
        # naming setting
        self.model_name = model_name
        
        # model setting
        self.input_shape = input_shape
        self.output_shape = output_shape
        with tf.variable_scope(self.model_name):
            self.x = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t = tf.placeholder(dtype=tf.float32, shape=self.output_shape, name='output')
            self.y = self._forward_pass(self.x)

    def _forward_pass(self, x):
        # Encoder
        h1 = conv_layer(x, filter_shape=[3, 3, 1, 64], name='L1') # (64, 64, 64)
        h2 = conv_layer(h1, filter_shape=[3, 3, 64, 64], name='L2')
        h3 = pooling_layer(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='L3') # (32, 32, 64)
        h4 = conv_layer(h3, filter_shape=[3, 3, 64, 128], name='L4')
        h5 = conv_layer(h4, filter_shape=[3, 3, 128, 128], name='L5')
        h6 = pooling_layer(h5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='L6') # (16, 16, 128)
        h7 = conv_layer(h6, filter_shape=[3, 3, 128, 256], name='L7')
        h8 = conv_layer(h7, filter_shape=[3, 3, 256, 256], name='L8')
        h9 = pooling_layer(h8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='L9') # (8, 8, 256)
        h10 = conv_layer(h9, filter_shape=[3, 3, 256, 512], name='L10')
        h11 = conv_layer(h10, filter_shape=[3, 3, 512, 512], name='L11')
        # Decoder
        #h12 = deconv_layer(h11, filter_shape=[3, 3, 512, 512], strides=[1, 2, 2, 1], output_shape=[-1, 128, 128, 512], name='L12')
        h12 = deconv_layer(h11, filter_shape=[3, 3, 256, 512], strides=[1, 2, 2, 1], output_shape=[-1, 16, 16, 256], name='L12')
        h12 = tf.concat([h12, h8], axis=3)
        h13 = conv_layer(h12, filter_shape=[3, 3, 512, 256], name='L13')
        h14 = conv_layer(h13, filter_shape=[3, 3, 256, 256], name='L14')
        #h15 = deconv_layer(h14, filter_shape=[3, 3, 256, 256], strides=[1, 2, 2, 1], output_shape=[-1, 256, 256, 256], name='L15')
        h15 = deconv_layer(h14, filter_shape=[3, 3, 128, 256], strides=[1, 2, 2, 1], output_shape=[-1, 32, 32, 128], name='L15')
        h15 = tf.concat([h15, h5], axis=3)
        h16 = conv_layer(h15, filter_shape=[3, 3, 256, 128], name='L16')
        h17 = conv_layer(h16, filter_shape=[3, 3, 128, 128], name='L17')
        #h18 = deconv_layer(h17, filter_shape=[3, 3, 128, 128], strides=[1, 2, 2, 1], output_shape=[-1, 512, 512, 128], name='L18')
        h18 = deconv_layer(h17, filter_shape=[3, 3, 64, 128], strides=[1, 2, 2, 1], output_shape=[-1, 64, 64, 64], name='L18')
        h18 = tf.concat([h18, h2], axis=3)
        h19 = conv_layer(h18, filter_shape=[3, 3, 128, 64], name='L19')
        h20 = conv_layer(h19, filter_shape=[3, 3, 64, 64], name='L20')
        h21 = conv_layer(h20, filter_shape=[1, 1, 64, self.input_shape[3]], name='L21', non_linear=None)
        return h21
    
    def optimize(self, loss):
        self.loss = loss(self.y, self.t)
        self.training = tf.train.AdamOptimizer(self.LR).minimize(self.loss)


class WNet(UNet):
    def __init__(self, LR, input_shape, output_shape, model_name='W-Net'):
        # optimization setting
        self.LR = LR
        
        # naming setting
        self.model_name = model_name
        
        # model setting
        self.input_shape = input_shape
        self.output_shape = output_shape
        with tf.variable_scope(self.model_name):
            self.x = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t = tf.placeholder(dtype=tf.float32, shape=self.output_shape, name='output')
            with tf.variable_scope('Encoder'):
                self.y_seg_logits = self._forward_pass(self.x)
                self.y_seg = tf.nn.sigmoid(self.y_seg_logits)
            with tf.variable_scope('Decoder'):
                self.y_rec_logits = self._forward_pass(self.y_seg)
                self.y_rec = tf.nn.sigmoid(self.y_rec_logits)
    
    def optimize(self, loss_seg, loss_rec):
        self.loss_seg = loss_seg(self.y_seg, self.t)
        self.loss_rec = loss_rec(self.y_rec_logits, self.t)
        self.training_enc = tf.train.AdamOptimizer(self.LR).minimize(self.loss_seg)
        self.training_dec = tf.train.AdamOptimizer(self.LR).minimize(self.loss_rec)
    