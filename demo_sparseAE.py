from models import SparseAE
from utils import increase_batch, mini_batch
from utils import image_crop
from sklearn.externals import tifffile

import tensorflow as tf
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
    
dataset = tifffile.imread('./Datase/carbon_fiber.tif')
x_train = dataset[200:801]
x_train = x_train[:, 370:640, 50:950]




sae = SparseAE(LR=1e-4, input_shape=[None, x_train.shape[1], x_train.shape[2], 1], 
            output_shape=[None, x_train.shape[1], x_train.shape[2], 1], )
def mse(predict, target):
    return tf.reduct_mean(tf.reduce_sum((predict-target)**2, axis=[1, 2, 3]))
sae.optimize(mse)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=5)




epoch = 30
batch_size = increase_batch(start=1, bound=5, rate=1e-3)
for ep in range(epoch):
    count = 0
    total_loss = 0
    start = time.time()
    for x, t in mini_batch(x_train, t_train, batch_generator=batch_size):
        count += 1
        feed_dict = {
            sae.x: x,
            sae.t: x,
        }
        loss = sess.run([sae.loss, sae.training], feed_dict=feed_dict)
        total_loss += loss
    end = time.time()
    message = "Epoch: {:<4} Loss: {:<10.9f} Time: {:<10.2f} "
    print(message.format(ep, total_loss/count, end-start))
save_path = saver.save(sess, './Models/sparseAE/sparseAE.ckpt')
