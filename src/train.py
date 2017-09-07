import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

# Include additional module
include_path = '../tensorflow_oop'
if include_path not in sys.path:
    sys.path.append(include_path)
from tensorflow_oop.regression import *

# Define model
class TFWeatherForecast(TFRegressor):
    def inference(self, inputs, **kwargs):
        # Get parameters
        input_size = self.inputs_shape[0]
        hidden_size = kwargs['hidden_size']
        rnn_count = kwargs['rnn_count']
        output_size = self.outputs_shape[-1]

        # Create RNN cells
        def rnn_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        cells = [rnn_cell() for _ in range(rnn_count)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        
        # Stack RNN layers
        batch_size = tf.shape(inputs)[0]
        init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)
        rnn_outputs, last_state = tf.nn.dynamic_rnn(cell=multi_cell,
                                                    initial_state=init_state,
                                                    inputs=inputs,
                                                    swap_memory=True,
                                                    time_major=False)

        # Output layer
        with tf.variable_scope('output'):
            W = tf.Variable(tf.truncated_normal([hidden_size, output_size]),
                            name='weight')
            b = tf.Variable(tf.zeros([output_size]),
                            name='bias')
            outputs = tf.nn.xw_plus_b(last_state[-1].h, W, b)
        return tf.reshape(outputs, [-1] + self.outputs_shape)

def run(args):
    print('Loading dataset...')
    dataset = TFDataset.load(args.input)
    print('Dataset shape: %s' % dataset.str_shape())

    # Configure batch size
    dataset.set_batch_size(args.batch_size)

    print('Splitting...')
    train_set, val_set, test_set = dataset.split(args.train_rate,
                                                 args.val_rate,
                                                 args.test_rate,
                                                 shuffle=args.shuffle)
    print('Traininig  set shape: %s' % train_set.str_shape())
    print('Validation set shape: %s' % val_set.str_shape())
    print('Testing    set shape: %s' % test_set.str_shape())

    print('Initializing...')
    model = TFWeatherForecast(log_dir=args.log_dir)
    model.initialize(inputs_shape=dataset.data_shape,
                     targets_shape=dataset.labels_shape,
                     outputs_shape=dataset.labels_shape,
                     hidden_size=args.hidden_size,
                     rnn_count=args.rnn_count)
    print('%s\n' % model)

    # Fitting model
    model.fit(train_set, epoch_count=args.epoch_count, val_set=val_set)

    print('Evaluating...')
    if train_set is not None:
        train_eval = model.evaluate(train_set)
        print('Results on training set: %s' % train_eval)
    if val_set is not None:
        val_eval = model.evaluate(val_set)
        print('Results on validation set: %s' % val_eval)
    if test_set is not None:
        test_eval = model.evaluate(test_set)
        print('Results on testing set: %s' % test_eval)

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
