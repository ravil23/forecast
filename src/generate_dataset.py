import argparse
import pandas as pd
import numpy as np
import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

# Include additional module
include_path = '../include'
if include_path not in sys.path:
    sys.path.append(include_path)
from tensorflow_oop import *

parser = argparse.ArgumentParser(description='Generate dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', '-i', type=str, required=True,
    help='path to raw data in csv format')
parser.add_argument('--output', '-o', type=str, required=True,
    help='path to saving dump of dataset')
parser.add_argument('--feature_names', type=str, nargs='+', required=True,
    help='column names for adding to dataset')
parser.add_argument('--interpolate', default=False, action='store_true',
    help='interpolate nan values')
parser.add_argument('--normalize', default=False, action='store_true',
    help='normalize features to zero mean and one std')
parser.add_argument('--seq_length', default=32, type=int,
    help='sequence length')
parser.add_argument('--seq_step', default=32, type=int,
    help='sequence iteration step')
parser.add_argument('--label_length', default=1, type=int,
    help='prediction label length')
parser.add_argument('--label_offset', default=0, type=int,
    help='offset of label after sequence end')

args = parser.parse_args()

def clean_file(filename):
    """
    Clean historical data file and write result to new file.
    Input:
        filename : string
    Output:
        cleaned_filename : string
    """
    # Parse input file
    with open(filename) as f:
        lines = []
        for line in f:
            # Skip comment lines
            if line[0] != '#':
                line = line.replace('\n', '')
                line = line.replace('\r', '')
                if line[-1] == ';':
                    # Delete last empty delimiter
                    line = line[:-1]
                lines.append(line + '\n')

    # Write result to new file
    cleaned_filename = filename + '.cleaned.csv'
    with open(cleaned_filename, 'w') as f:
        f.writelines(lines)

    return cleaned_filename

def load_dataframe(filename):
    """
    Load dataframe from csv.
    Input:
        filename : string
    Output:
        df : pandas.core.frame.DataFrame
    """
    # Read file
    df = pd.read_csv(filename, delimiter=';', quotechar='"')

    # Convert string to datetime format
    df['time'] = pd.to_datetime(df.iloc[:,0], format='%d.%m.%Y %H:%M')

    # Set time as index
    df = df.set_index('time')

    # Resample with timedelta 3 hours and fill new rows with NaN
    df = df.resample('3H').asfreq()

    # Preprocessing amount of precipitation
    df['RRR'] = df['RRR'].replace('No precipitation', 0.)
    df['RRR'] = df['RRR'].replace('Trace of precipitation', np.nan)
    df['RRR'] = pd.to_numeric(df['RRR'])
    return df

if __name__ == '__main__':
    print 'Cleaning...'
    cleaned_filename = clean_file(args.input)
    print 'Cleaned filename:', cleaned_filename, '\n'

    print 'Loading...'
    df = load_dataframe(cleaned_filename)
    print 'Loaded dataframe shape:', df.shape, '\n'

    print 'Getting features...'
    features = TFDataframe(df[args.feature_names])
    print 'Features shape:', features.df_.shape, '\n'

    if args.interpolate:
        print 'Interpolating...'
        nan_count = features.df_.isnull().sum()
        features.interpolate()
        print 'Interpolated NaN values count:\n', nan_count, '\n'

    if args.normalize:
        print 'Normalizing...'
        features.normalize()
        print 'Normalized with parameters:'
        print '\tmean:\n', features.normalization_mean_
        print '\tstd:\n', features.normalization_std_, '\n'

    print 'Generating dataset...'
    dataset = features.generate_sequences(args.seq_length, args.seq_step,
                                          args.label_length, args.label_offset)
    print dataset, '\n'

    print 'Saving dataset...'
    dataset.save(args.output)
    print 'Dataset saved to:', args.output
