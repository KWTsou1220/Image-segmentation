from models import SparseAE
from utils import increase_batch, mini_batch
from utils import image_crop
from skimage.external import tifffile
from skimage import img_as_float32

import skimage
import tensorflow as tf
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
    
dataset = tifffile.imread('./Dataset/carbon_fiber.tif')
x_train = dataset[200:801]
x_train = x_train[:, 370:640, 50:950]
x_train = np.expand_dims(x_train, axis=3)
x_train = img_as_float32(x_train)
print('Reading dataset is done!')



sae = SparseAE(LR=1e-4, input_shape=[None, x_train.shape[1], x_train.shape[2], 1], 
            output_shape=[None, x_train.shape[1], x_train.shape[2], 1], )
def mse(predict, target):
    return tf.reduce_mean(tf.reduce_sum((predict-target)**2, axis=[1, 2, 3]))
sae.optimize(mse)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=5)



print('Training...')
epoch = 30
batch_size = increase_batch(start=1, bound=50, rate=1e-3)
for ep in range(epoch):
    count = 0
    total_loss = 0
    start = time.time()
    for x, t in mini_batch(x_train, x_train, batch_generator=batch_size):
        count += 1
        feed_dict = {
            sae.x: x,
            sae.t: x,
            sae.is_train: True,
        }
        loss, _ = sess.run([sae.loss, sae.training], feed_dict=feed_dict)
        total_loss += loss
    end = time.time()
    message = "Epoch: {:<4} Loss: {:<10.9f} Time: {:<10.2f} "
    print(message.format(ep, total_loss/count, end-start))
save_path = saver.save(sess, './Models/sparseAE/sparseAE.ckpt')



print('Testing...')
epoch = 1
batch_size = increase_batch(start=50, bound=50, rate=0)
detect_img = []
code_img = []
reconstr_img = []
for ep in range(epoch):
    start = time.time()
    for x, t in mini_batch(x_train, x_train, batch_generator=batch_size):
        feed_dict = {
            sae.x: x,
            sae.t: x,
            sae.is_train: False,
        }
        detect, code, reconstr = sess.run([sae.detect_c, sae.c, sae.y], feed_dict=feed_dict)
        detect_img += [detect]
        code_img += [code]
        reconstr_img += [reconstr]
    end = time.time()
    message = "Epoch: {:<4} Loss: {:<10.9f} Time: {:<10.2f} "
    print(message.format(ep, total_loss/count, end-start))

detect_img = np.stack(detect_img, axis=0)
code_img = np.stack(code_img, axis=0)
reconstr_img = np.stack(reconstr_img, axis=0)

tifffile.imsave('./Dataset/prediction/detect_img.tif', detect_img)
tifffile.imsave('./Dataset/prediction/code_img.tif', code_img)
tifffile.imsave('./Dataset/prediction/reconstr_img.tif', reconstr_img)

