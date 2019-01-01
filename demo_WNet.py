from models import WNet
from utils import increase_batch, mini_batch
from utils import read_EM, EM_image_cut
from utils import soft_ncut, sigmoid_loss

import tensorflow as tf
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
import os
    
x_train, t_train, x_test = read_EM("./Dataset/")
x_train = EM_image_cut(x_train, 64)
t_train = EM_image_cut(t_train, 64)
x_test = EM_image_cut(x_test, 64)

wnet = WNet(LR=1e-8, input_shape=[None, x_train.shape[1], x_train.shape[2], 1], 
            output_shape=[None, t_train.shape[1], t_train.shape[2], 1], )
wnet.optimize(soft_ncut, sigmoid_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

epoch = 100
batch_size = increase_batch(start=1, bound=15, rate=1e-3)
for ep in range(epoch):
    count = 0
    total_seg_loss = 0
    total_rec_loss = 0
    start = time.time()
    for x, t in mini_batch(x_train, t_train, batch_generator=batch_size):
        count += 1
        feed_dict = {
            wnet.x: np.expand_dims(x, axis=3),
            wnet.t: np.expand_dims(x, axis=3),
        }
        loss_seg, _ = sess.run([wnet.loss_seg, wnet.training_enc], feed_dict=feed_dict)
        loss_rec, _ = sess.run([wnet.loss_rec, wnet.training_dec], feed_dict=feed_dict)
        total_seg_loss += loss_seg
        total_rec_loss += loss_rec
    end = time.time()
    message = "Epoch: {:<4} Segmentation Loss: {:<12.9f} Reconstruction Loss: {:<10.9f} Time: {:<10.2f} "
    print(message.format(ep, total_seg_loss/count, total_rec_loss/count, end-start))
    if not os.path.isdir('./Models/wnet'+str(ep)):
        os.mkdir('./Models/wnet'+str(ep))
    save_path = saver.save(sess, './Models/wnet'+str(ep)+'/wnet.ckpt')
    if ep==0 and os.path.isfile('./log'):
        os.remove('./log')
    with open('./log', 'a') as file_write:
        file_write.write(message.format(ep, total_seg_loss/count, total_rec_loss/count, end-start))
        file_write.write('\n')

