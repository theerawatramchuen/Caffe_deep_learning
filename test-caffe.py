#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:28:34 2019

@author: theerawatra
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import caffe

#PATHS = '/home/theerawatra/caffe/examples/mnist/'
PATHS = ''

MODEL_FILE = PATHS + 'lenet.prototxt'
PRETRAINED = PATHS + 'lenet_iter_10000.caffemodel'
IMAGE_FILE = PATHS + '33.jpeg'

input_image = caffe.io.load_image(IMAGE_FILE, color=False)
net = caffe.Classifier(MODEL_FILE, PRETRAINED)


prediction = net.predict([input_image], oversample = False)
caffe.set_mode_gpu()
print ('predicted class:', prediction[0].argmax())
