import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

# Include additional module
include_path = '../include'
if include_path not in sys.path:
    sys.path.append(include_path)
from tensorflow_oop import *

# Define model
class TFWeatherForecast(TFRegressor):
    def inference(self, inputs, **kwargs):
        input_size = self.inputs_shape_[0]
        hidden_size = kwargs['hidden_size']
        rnn_count = kwargs['rnn_count']
        output_size = self.outputs_shape_[0]
        def rnn_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        cells = [rnn_cell() for _ in xrange(rnn_count)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        batch_size = tf.shape(inputs)[0]
        init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)
        rnn_outputs, last_state = tf.nn.dynamic_rnn(cell=multi_cell,
                                                    initial_state=init_state,
                                                    inputs=inputs,
                                                    time_major=False)
        W = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
        b = tf.Variable(tf.zeros([output_size]))
        outputs = tf.nn.xw_plus_b(rnn_outputs[:,-1], W, b)
        return tf.reshape(outputs, [-1] + self.outputs_shape_)

def run(args):
    print 'Loading dataset...'
    dataset = TFDataset()
    dataset.load(args.input)
    print 'Dataset shape:', dataset.size_, dataset.data_shape_, '->', dataset.labels_shape_, '\n'

    # Configure batch size
    dataset.set_batch_size(args.batch_size)

    print 'Splitting...'
    train_set, val_set, test_set = dataset.split(args.train_rate, args.val_rate, args.test_rate, shuffle=args.shuffle)
    print 'Traininig  set shape:', train_set.size_, train_set.data_shape_, '->', train_set.labels_shape_
    print 'Validation set shape:', val_set.size_, val_set.data_shape_, '->', val_set.labels_shape_
    print 'Testing    set shape:', test_set.size_, test_set.data_shape_, '->', test_set.labels_shape_, '\n'

    print 'Initializing...'
    model = TFWeatherForecast(log_dir=args.log_dir,
                              inputs_shape=dataset.data_shape_,
                              outputs_shape=dataset.labels_shape_,
                              hidden_size=args.hidden_size,
                              rnn_count=args.rnn_count)
    print model, '\n'

    # Fitting model
    model.fit(train_set, args.epoch_count, val_set=val_set)

    print 'Evaluating...'
    if train_set is not None:
        train_eval = model.evaluate(train_set)
        print 'Results on training set:', train_eval
    if val_set is not None:
        val_eval = model.evaluate(val_set)
        print 'Results on validation set:', val_eval
    if test_set is not None:
        test_eval = model.evaluate(test_set)
        print 'Results on testing set:', test_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', type=str, required=True,
        help='path to dump of dataset')
    parser.add_argument('--log_dir', type=str, required=True,
        help='path to directory for logging')
    parser.add_argument('--batch_size', default=32, type=int,
        help='batch size')
    parser.add_argument('--train_rate', default=0.7, type=float,
        help='training set rate for dataset splitting')
    parser.add_argument('--val_rate', default=0.15, type=float,
        help='validation set rate for dataset splitting')
    parser.add_argument('--test_rate', default=0.15, type=float,
        help='testing set rate for dataset splitting')
    parser.add_argument('--shuffle', default=False, action='store_true',
        help='randomize dataset elements order before splitting')
    parser.add_argument('--hidden_size', default=100, type=int,
        help='hidden layer size')
    parser.add_argument('--rnn_count', default=1, type=int,
        help='rnn layers count')
    parser.add_argument('--epoch_count', default=1000, type=int,
        help='training epoch count')
    parser.add_argument('--learning_rate', default=0.001, type=float,
        help='learning rate')

    args = parser.parse_args()

    run(args)
