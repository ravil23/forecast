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

def run(args):
    print('Loading model...')
    model = TFRegressor.load(args.model)
    print('%s\n' % model)

    history = np.zeros(model.inputs_shape_)

    print('Predicting...')
    while True:
        # Forward propagation
        predict = model.forward(history)
        print('Predict: %s' % predict)

        # Update history
        history[:-1] = history[1:]
        history[-1] = predict
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online predict weather forecast',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, required=True,
        help='path to model')

    args = parser.parse_args()

    run(args)
