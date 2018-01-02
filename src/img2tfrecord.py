#-*-coding:utf-8 -*-

# For generating tfrecords for the new HWDB-text data.
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import tensorflow as tf
import numpy as np
import math
from PIL import Image  
import matplotlib.pyplot as plt

DATA_DIR = '/home/tal-cai/Src/cnn_lstm_hwdb_ocr/data/image'
OUTPUT_DIR = '/home/tal-cai/Src/cnn_lstm_hwdb_ocr/data/'
CHAR_SET_DIR = '/home/tal-cai/Src/cnn_lstm_hwdb_ocr/src/dict'

kernel_sizes = [5,5,3,3,3,3]

png_data = tf.placeholder(dtype=tf.string)
png_decoder = tf.image.decode_png(png_data,channels=1)

def get_charset():
	dict_file = open(CHAR_SET_DIR,'r')
	temp_set = []
	for line in dict_file:
		char = line.strip().split()[0]
		temp_set.append(char)
	return temp_set
ch_charset = get_charset()
	

def calc_seq_len(image_width):
    """Calculate sequence length of given image after CNN processing"""
    
    conv1_trim =  2 * (kernel_sizes[0] // 2)
    fc6_trim = 2*(kernel_sizes[5] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool1 = after_conv1 // 2
    after_pool2 = after_pool1 // 2
    after_pool4 = after_pool2 - 1 # max without stride
    after_fc6 =  after_pool4 - fc6_trim
    seq_len = 2*after_fc6
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(1920)]	

def make_example(filename, image_data, labels, text, height, width):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature(labels),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'text/string': _bytes_feature(tf.compat.as_bytes(text)),
        'text/length': _int64_feature([len(text)])
    }))
    return example

def generate_tfrecord(image_base_dir,output_dir,train_ratio = 0.95):
	train_count = 0
	test_count = 0
	session_config = tf.ConfigProto()
	session_config.gpu_options.allow_growth=True
	sess = tf.Session(config=session_config)
	image_filenames = get_image_list(image_base_dir)

	for image in image_filenames:
		image_dir = os.path.join(image_base_dir,image)
		image_data, img_height, img_width = get_image(sess,image_dir)
		text,label = get_text_and_labels(image)
#		print text,label,image_data.shape
#		plt.imshow(image_data[:,:,0],cmap='gray')
#		plt.show()
		train_dir = ''
		out_put_filename = ''
		if np.random.randint(1, 100) < 100 * train_ratio:
			train_dir = 'train'
			out_put_filename = train_dir + '-' + str(train_count) + '.tfrecord'
			train_count += 1
		else:
			train_dir = 'test'
			out_put_filename = train_dir + '-' + str(test_count) + '.tfrecord'
			test_count += 1
		output_dir = os.path.join(OUTPUT_DIR,train_dir,out_put_filename)
		example = make_example(image, image_data, label, text, img_height, img_width)
		try:
			writer = tf.python_io.TFRecordWriter(output_dir)
			writer.write(example.SerializeToString())
			print '{0} generated.'.format(out_put_filename)
		except:
			print 'ERROR',out_put_filename
		
def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    image = sess.run(png_decoder,feed_dict={png_data: image_data})
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width

def get_text_and_labels(filename):
    """ Extract the human-readable text and label sequence from image filename"""
    text = os.path.basename(filename.decode('utf-8')).split('_')[0]
    labels = [ch_charset.index(c) for c in list(text)]
    return text,labels

def get_image_list(image_base_dir):
	return os.listdir(image_base_dir)

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


if __name__ == '__main__':
	generate_tfrecord(DATA_DIR,OUTPUT_DIR)
	# print seq_lens[256]

























