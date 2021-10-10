# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: test_head_count.py
@time: 19-1-4 09:44
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import time
import cv2
import numpy as np

import sys
sys.path.append('../')
from src.lightweight_openpose import lightweight_openpose
from src.pose_decode import decode_pose



params = {}
params['test_model'] = './model/model.ckpt-61236'
params['thre1'] = 0.1
params['thre2'] = 0.0

def main():

    use_gpu = False

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    input_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # _1, _2, cpm, paf = light_openpose(input_img, is_training=False)
    cpm, paf = lightweight_openpose(input_img, num_pafs=26, num_joints=14, is_training=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, params['test_model'])
        print('#---------Successfully loaded trained model.---------#')

        camera = cv2.VideoCapture(0)

        while True:
            time_start = time.time()
            retval, img_ori = camera.read()
            if not retval:
                break

            img_data = cv2.cvtColor(img_ori, code=cv2.COLOR_BGR2RGB)
            img_data = cv2.resize(img_data, (256, 256))
            img = img_data / 255.

            heatmap, _paf = sess.run([cpm, paf], feed_dict={input_img: [img]})

            canvas, joint_list, person_to_joint_assoc, joints = decode_pose(img_data, params, heatmap[0], _paf[0])

            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            cv2.imshow('result', canvas)

            time_end = time.time()
            FPS = 1 / (time_end - time_start)
            print('FPS = ', FPS)


            if cv2.waitKey(1) == ord('q'):
                break

main()