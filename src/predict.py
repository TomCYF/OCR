# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import textsyn
import model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('output','predict',
                          """Sub-directory of model for test summary events""")

tf.app.flags.DEFINE_integer('size',1,
                            """predict data size""")
tf.app.flags.DEFINE_integer('test_interval_secs', 60,
                             'Time between test runs')

tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")

tf.app.flags.DEFINE_string('file_path','../data/predict/testdemo2.png',
                           """Base directory for ../data/predict data""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers

CHAR_SET_DIR = '/home/tal-cai/Src/cnn_lstm_hwdb_ocr/src/dict'

import codecs
def get_charset():
	dict_file = codecs.open(CHAR_SET_DIR,'r',encoding='utf-8')
	temp_set = []
	for line in dict_file:
		char = line.strip().split()[0]
		temp_set.append(char)
	return temp_set
ch_charset = get_charset()
print len(ch_charset)

def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def _get_input():
	""" Load the image from source"""
	
	raw_image = tf.gfile.FastGFile(FLAGS.file_path, 'r').read()  
	image_data = tf.image.decode_png(raw_image)
	image_data = tf.image.convert_image_dtype(image_data,dtype=tf.uint8)
#	image_data = tf.image.flip_up_down(image_data)
#	image_data = tf.image.transpose_image(image_data)
	image_data = tf.image.rgb_to_grayscale(image_data)

#	new_height = tf.constant(32, dtype=tf.int32)
#	shape = tf.shape(image_data)
#	height = shape[0]
#	width = shape[1]
	
#	new_width =  (new_height / height) * width
#	print shape,height,width,new_height,new_width
	image_data = tf.image.resize_images(image_data,[64,576])
#	image_shape = image_data.get_shape()
#	new_width = image_shape[0]/32.*image_shape[1]
#	print image_shape#,new_width
#	image_data = tf.image.resize_images(image_data,(32,new_width))

#	height = image_data.get_shape()[0]
#   width = image_data.get_shape()[1]
	
	return image_data

def _get_predict(rnn_logits, sequence_length):
	with tf.name_scope("predict"):
		predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=2,
                                                   merge_repeated=False)
		hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
	return hypothesis.values

def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def main(argv=None):
	
	with tf.Graph().as_default():
		image = _get_input()
	
		with tf.device(FLAGS.device):
			pred_img = tf.placeholder(tf.float32,[1,64,None,1])
		 	img_width = tf.placeholder(tf.int32)
		 	features,sequence_length = model.convnet_layers( pred_img, img_width, mode)
		 	logits = model.rnn_layers( features, sequence_length,
                                        textsyn.num_classes())
		 	recognition_hypothesis = _get_predict(logits, sequence_length)
		
		session_config = _get_session_config()
		restore_model = _get_init_trained()
		
		init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        
		step_ops = [recognition_hypothesis,logits,sequence_length]
	
		with tf.Session(config=session_config) as sess:
			sess.run(init_op)
			
			image_data = sess.run(image)
			image_data=image_data.reshape((64,576))
			restore_model(sess, _get_checkpoint())
			# image_data = plt.imread(FLAGS.file_path)
			# image_data = rgb2gray(image_data)
#			img_height = image_data.shape[0]
#			img_width = image_data.shape[1]
#			new_width = int( (32./img_height) * img_width)
			# image_data.resize((32,int((32.0/img_height)*img_width)))
			
			# data_width = image_data.shape[1]
			
			
			
			print image_data.shape
			plt.imshow(image_data, cmap = plt.cm.gray)
			
			image_data = image_data.reshape((1,64,576,1))
			pred_output,l,sl = sess.run(step_ops,feed_dict={pred_img:image_data,img_width:576})

			print pred_output
			text_output = [ch_charset[c].encode('utf-8') for c in pred_output]
			text = ""
			for i in text_output:
				text += i.encode('utf-8')
			print text
			print sl
			#for j in l:
				#print ch_charset[np.argmax(j)-1].encode('utf-8'),
				#print j,			
			plt.title(text)		
			plt.show()
			

if __name__ == '__main__':
	tf.app.run()





