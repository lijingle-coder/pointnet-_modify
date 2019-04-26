from tensorflow.python.platform import gfile

'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_dataset
import modelnet_h5_dataset
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 2
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()
num_votes=1
# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=4096)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

sess = tf.Session()
with gfile.FastGFile('./good_frozen.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图
 
pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
is_training_pl = tf.placeholder(tf.bool, shape=())
# 需要有一个初始化的过程    
sess.run(tf.global_variables_initializer())
output = sess.graph.get_tensor_by_name("fc3/BiasAdd:0")
input = sess.graph.get_tensor_by_name("Placeholder:0")
input_1 = sess.graph.get_tensor_by_name("Placeholder_1:0")
input_2 = sess.graph.get_tensor_by_name("Placeholder_2:0")





# Make sure batch data is of same size
cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

total_correct = 0
total_seen = 0
loss_sum = 0
batch_idx = 0
shape_ious = []
total_seen_class = [0 for _ in range(NUM_CLASSES)]
total_correct_class = [0 for _ in range(NUM_CLASSES)]

while TEST_DATASET.has_next_batch():
	batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
	bsize = batch_data.shape[0]
	print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
	# for the last batch in the epoch, the bsize:end are from last batch
	cur_batch_data[0:bsize,...] = batch_data
	cur_batch_label[0:bsize] = batch_label

	batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
	for vote_idx in range(num_votes):
		# Shuffle point order to achieve different farthest samplings
		shuffled_indices = np.arange(NUM_POINT)
		np.random.shuffle(shuffled_indices)
		if FLAGS.normal:
			rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
				vote_idx/float(num_votes) * np.pi * 2)
		else:
			rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
				vote_idx/float(num_votes) * np.pi * 2)
		is_training = False
		#feed_dict = {input: rotated_data,input_1: cur_batch_label,input_2: is_training}
		
		pred_val = sess.run(output, feed_dict={input:cur_batch_data[:, shuffled_indices, :],input_2:is_training})
		print(pred_val)

class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
for i, name in enumerate(SHAPE_NAMES):
	log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


 

# 输出 26
