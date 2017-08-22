import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
import math
import os
import time
import pickle
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import device_lib

# Set logging level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TFHelper:

    """
    Helpful functions.
    """

    @staticmethod
    def devices_list():
        """List of available devices."""
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    @staticmethod
    def checkpoints_list(log_dir):
        """List of all model checkpoint paths."""
        checkpoint_state = tf.train.get_checkpoint_state(log_dir)
        return checkpoint_state.all_model_checkpoint_paths

class TFBatch:

    """
    Batch container.
    """

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        string = "TFBatch object:\n"
        for attr in self.__dict__:
            string += "%s: \n%s\n" % (attr, getattr(self, attr))
        return string[:-1]

def check_initialization(function):
    """Decorator for check initialization."""
    def wrapper(self, *args, **kwargs):
        assert self.init_, \
            'Object should be initialized: self.init_ = %s' % self.init_
        return function(self, *args, **kwargs)
    return wrapper

def check_inputs_values(function):
    """Decorator for check corresponding inputs values."""
    def wrapper(self, inputs_values, *args, **kwargs):
        assert np.all(np.asarray(inputs_values.shape[1:]) == np.asarray(self.inputs_shape_)), \
            'Inputs values shape should be correspond to model inputs shape: inputs_values.shape = %s, self.inputs_shape_ = %s' % (inputs_values.shape, self.inputs_shape_) 
        return function(self, inputs_values=inputs_values, *args, **kwargs)
    return wrapper        

class TFDataset:

    """
    Dataset structure.
    """

    __slots__ = ['init_', 'size_',
                 'data_', 'data_shape_', 'data_ndim_',
                 'labels_', 'labels_shape_', 'labels_ndim_',
                 'batch_size_', 'batch_count_', 'batch_num_',
                 'normalized_', 'normalization_mask_', 'normalization_mean_', 'normalization_std_']

    def __init__(self, data=None, labels=None):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.init_ = False
        self.size_ = 0
        self.batch_size_ = 1
        self.batch_num_ = 0
        if data is not None or labels is not None:
            self.initialize(data=data, labels=labels)

    def copy(self, other):
        """Copy other dataframe."""
        for attr in self.__slots__:
            setattr(self, attr, getattr(other, attr))

    def initialize(self, data, labels):
        """Set data and labels."""
        assert data is not None or labels is not None, \
            'Data or labels should be passed: data = %s, labels = %s' % (data, labels)
        if data is not None and labels is not None:
            assert len(data) == len(labels), \
                'Data and labels should be the same length: len(data) = %s, len(labels) = %s' % (len(data), len(labels))
        if data is not None:
            self.size_ = len(data)
            data = np.asarray(data)
            if data.ndim == 1:
                self.data_ = np.reshape(data, (self.size_, 1))
            else:
                self.data_ = data
            self.data_shape_ = list(self.data_.shape[1:])
            self.data_ndim_ = len(self.data_shape_)
        else:
            self.data_ = None
            self.data_shape_ = None
            self.data_ndim_ = None
            self.normalized_ = None
            self.normalization_mask_ = None
            self.normalization_mean_ = None
            self.normalization_std_ = None
        if labels is not None:
            self.size_ = len(labels)
            labels = np.asarray(labels)
            if labels.ndim == 1:
                self.labels_ = np.reshape(labels, (self.size_, 1))
            else:
                self.labels_ = labels
            self.labels_shape_ = list(self.labels_.shape[1:])
            self.labels_ndim_ = len(self.labels_shape_)
        else:
            self.labels_ = None
            self.labels_shape_ = None
            self.labels_ndim_ = None
        self.batch_count_ = int(self.size_ / self.batch_size_)
        self.init_ = True

    @check_initialization
    def shuffle(self):
        """Random shuffling of dataset."""
        indexes = np.arange(self.size_)
        np.random.shuffle(indexes)
        self.data_ = self.data_[indexes]
        self.labels_ = self.labels_[indexes]

    @check_initialization
    def set_batch_size(self, batch_size):
        """Set batch size."""
        assert batch_size > 0, \
            'Batch size should be greater then zero: batch_size = %s' % batch_size
        assert batch_size <=  self.size_, \
            'Batch size should not be greater then dataset size: batch_size = %s, self.size_ = %s' % (batch_size, self.size_)
        self.batch_size_ = int(batch_size)
        self.batch_count_ = int(self.size_ / self.batch_size_)

    @check_initialization
    def next_batch(self):
        """Get next batch."""
        first = (self.batch_num_ * self.batch_size_) % self.size_
        last = first + self.batch_size_
        batch_data = None
        batch_labels = None
        if (last <= self.size_):
            if self.data_ is not None:
                batch_data = self.data_[first:last]
            if self.labels_ is not None:
                batch_labels = self.labels_[first:last]
        else:
            if self.data_ is not None:
                batch_data = np.append(self.data_[first:], self.data_[:last - self.size_], axis=0)
            if self.labels_ is not None:
                batch_labels = np.append(self.labels_[first:], self.labels_[:last - self.size_], axis=0)
        self.batch_num_ += 1
        return TFBatch(data_=batch_data, labels_=batch_labels)

    @check_initialization
    def iterbatches(self, count=None):
        """Get iterator by batches."""
        if count is None:
            count = self.batch_count_
        for i in range(count):
            yield self.next_batch()

    @check_initialization
    def split(self, train_size, val_size, test_size, shuffle):
        """Split dataset to train, validation and test set."""
        assert train_size >= 0, \
            'Training size should not be less then zero: train_size = %s' % train_size
        assert val_size >= 0, \
            'Validation size should not be less then zero: val_size = %s' % val_size
        assert test_size >= 0, \
            'Testing size should not be less then zero: test_size = %s' % test_size
        total_size = train_size + val_size + test_size
        assert total_size == self.size_ or total_size == 1, \
            'Total size should be equal to TFDataset size or one: total_size = %s, self.size_ = %s' % (total_size, self.size_)
        if total_size == 1:
            if train_size != 0:
                train_size = int(round(float(train_size) * self.size_))
            if test_size != 0:
                if val_size != 0:
                    test_size = int(round(float(test_size) * self.size_))
                else:
                    test_size = self.size_ - train_size
            if val_size != 0:
                val_size = self.size_ - train_size - test_size
        indexes = np.arange(self.size_)
        if shuffle:
            np.random.shuffle(indexes)
        if train_size > 0:
            train_set = TFDataset()
            train_set.copy(self)
            data = self.data_[indexes[:train_size]] if self.data_ is not None else None
            labels = self.labels_[indexes[:train_size]] if self.labels_ is not None else None
            train_set.initialize(data, labels)
        else:
            train_set = None
        if val_size > 0:
            val_set = TFDataset()
            val_set.copy(self)
            data = self.data_[indexes[train_size:train_size + val_size]] if self.data_ is not None else None
            labels = self.labels_[indexes[train_size:train_size + val_size]] if self.labels_ is not None else None
            val_set.initialize(data, labels)
        else:
            val_set = None
        if test_size > 0:
            test_set = TFDataset()
            test_set.copy(self)
            data = self.data_[indexes[-test_size:]] if self.data_ is not None else None
            labels = self.labels_[indexes[-test_size:]] if self.labels_ is not None else None
            test_set.initialize(data, labels)
        else:
            test_set = None
        return train_set, val_set, test_set

    @staticmethod
    def load(filename):
        """Load dataset from file."""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, TFDataset), \
            'Loaded object should be TFDataset object: type(obj) = %s' % type(obj)
        return obj

    @check_initialization
    def save(self, filename):
        """Save dataset to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @check_initialization
    def generate_sequences(self, sequence_length, sequence_step, label_length=None, label_offset=None):
        """Generate sequences."""
        assert self.data_ is not None, \
            'Data field should be initialized: self.data_ = %s' % self.data_
        assert sequence_length > 0, \
            'Sequence length should be greater than zero: sequence_length = %s' % sequence_length
        assert sequence_step > 0, \
            'Sequence step should be greater than zero: sequence_step = %s' % sequence_step
        if label_length is not None:
            assert label_length > 0 and label_offset is not None, \
                'Label length should be greater than zero and label offset passed: label_length = %s, label_offset = %s' % (label_length, label_offset)
        if label_length is not None:
            sequences = []
            labels = []
            last = self.size_ - sequence_length - label_length - label_offset + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data_[i : last_sequence_index]
                sequences.append(current_sequence)
                first_label_index = last_sequence_index + label_offset
                current_label = self.data_[first_label_index : first_label_index + label_length]
                labels.append(current_label)
            dataset = TFDataset(sequences, labels)
            dataset.copy(self)
            dataset.initialize(data=sequences, labels=labels)
        else:
            sequences = []
            last = self.size_ - sequence_length + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data_[i : last_sequence_index]
                sequences.append(current_sequence)
            dataset = TFDataset(sequences)
            dataset.copy(self)
            dataset.initialize(data=sequences, labels=None)
        if self.normalization_mask_ is not None:
            dataset.normalization_mask_ = [False] + self.normalization_mask_
        return dataset

    @check_initialization
    def normalize(self, mask=None):
        """
        Normalize data to zero mean and one std by mask.
        Where mask is boolean indicators corresponding to data dimensions.
        If mask value is True, then feature with this dimension should be normalized.
        """
        assert self.data_ is not None, \
            'Data field should be initialized: self.data_ = %s' % self.data_
        if self.normalized_:
            return
        if mask is not None:
            assert len(mask) == self.data_ndim_, \
                'Mask length should be equal to data dimensions count: len(mask) = %s, self.data_ndim_ = %s' % (len(mask), self.data_ndim_)

            for i in range(0, len(mask) - 1):
                assert mask[i + 1] or not mask[i], \
                    'False elements should be before True elements: mask = %s' % mask

            assert mask[-1] == True, \
                'Last mask element should be True: mask = %s' % mask

            # Reshape to array of features
            data_shape_arr = np.asarray(self.data_shape_)
            new_shape = [-1] + list(data_shape_arr[mask])
            reshaped_data = np.reshape(self.data_, new_shape)

            # Save normalisation properties
            self.normalization_mask_ = list(mask)
            self.normalization_mean_ = np.mean(reshaped_data, axis=0)
            self.normalization_std_ = np.std(reshaped_data, axis=0)

            # Reshape normalization properties for correct broadcasting
            valid_shape = data_shape_arr
            valid_shape[np.logical_not(self.normalization_mask_)] = 1
            reshaped_normalization_mean_ = np.reshape(self.normalization_mean_, valid_shape)
            reshaped_normalization_std_ = np.reshape(self.normalization_std_, valid_shape)

            # Replace zero std with one
            valid_normalization_std_ = reshaped_normalization_std_
            valid_normalization_std_[reshaped_normalization_std_ == 0] = 1

            # Update dataset with normalized value
            self.data_ = (self.data_ - reshaped_normalization_mean_) / valid_normalization_std_
        else:
            # Save normalisation properties
            self.normalization_mask_ = None
            self.normalization_mean_ = np.mean(self.data_)
            self.normalization_std_ = np.std(self.data_)

            # Update dataset with normalized value
            self.data_ = (self.data_ - self.normalization_mean_) / self.normalization_std_
        self.normalized_ = True

    @check_initialization
    def unnormalize(self):
        """Unnormalize dataset to original from zero mean and one std."""
        assert self.data_ is not None, \
            'Data field should be initialized: self.data_ = %s' % self.data_
        if not self.normalized_:
            return
        if self.normalization_mask_ is not None:
            data_shape_arr = np.asarray(self.data_shape_)

            # Reshape for correct broadcasting
            valid_shape = data_shape_arr
            valid_shape[np.logical_not(self.normalization_mask_)] = 1
            reshaped_normalization_mean_ = np.reshape(self.normalization_mean_, valid_shape)
            reshaped_normalization_std_ = np.reshape(self.normalization_std_, valid_shape)

            # Replace zero std with one 
            valid_normalization_std_ = reshaped_normalization_std_
            valid_normalization_std_[reshaped_normalization_std_ == 0] = 1

            # Update dataset with unnormalized value
            self.data_ = self.data_ * valid_normalization_std_ +  reshaped_normalization_mean_
        else:
            # Update dataset with unnormalized value
            self.data_ =  self.data_ * self.normalization_std_ +  self.normalization_mean_
        self.normalized_ = False

    def __len__(self):
        return self.size_

    def __str__(self):
        string = 'TFDataset object:\n'
        for attr in self.__slots__:
            if attr != 'data_' and attr != 'labels_':
                string += "%20s: %s\n" % (attr, getattr(self, attr))
        if 'data_' in self.__slots__:
            string += "%s: \n%s\n" % ('data_', getattr(self, 'data_'))
        if 'labels_' in self.__slots__:
            string += "%s: \n%s\n" % ('labels_', getattr(self, 'labels_'))
        return string[:-1]

class TFTripletset(TFDataset):

    """
    Triplet generation dataset.
    """

    __slots__ = TFDataset.__slots__ + ['batch_positives_count_', 'batch_negatives_count_', 'labels_counts_']

    def initialize(self, data, labels):
        assert data is not None and labels is not None, \
            'Data and labels should be passed: data = %s, labels = %s' % (data, labels)
        ndim = np.asarray(labels).ndim
        assert ndim == 1, \
            'Labels should be 1D dimension: labels.ndim = %s' % ndim
        super().initialize(data, labels)

    @check_initialization
    def set_batch_size(self, batch_size, batch_positives_count):
        assert batch_positives_count > 0, \
            'Positives count in batch should be greater than zero: batch_positives_count = %s' % batch_positives_count
        assert batch_positives_count < batch_size, \
            'Positives count in batch should be less than batch size: batch_positives_count = %s, batch_size = %s' % (batch_positives_count, batch_size)
        super().set_batch_size(batch_size)
        self.batch_positives_count_ = int(batch_positives_count)
        self.batch_negatives_count_ = self.batch_size_ - self.batch_positives_count_

    @check_initialization
    def next_batch(self):
        labels = self.labels_.flatten()
        labels_counts = np.bincount(labels)
        positive_keys = np.where(labels_counts >= self.batch_positives_count_)[0]
        positive_key = positive_keys[np.random.randint(0, len(positive_keys))]

        # Take positive samples
        positives = self.data_[labels == positive_key]
        positives = positives[np.random.choice(np.arange(labels_counts[positive_key]), self.batch_positives_count_, replace=False), :]

        # Take negative samples
        negatives = self.data_[labels != positive_key]
        negatives = negatives[np.random.choice(np.arange(negatives.shape[0]), self.batch_negatives_count_, replace=False), :]

        batch_data = np.vstack([positives, negatives])
        batch_labels = np.append(np.zeros(len(positives)), np.ones(len(negatives)))
        return TFBatch(data_=batch_data, labels_=batch_labels)

class TFNeuralNetwork:

    """
    Basic neural network model.
    """

    __slots__ = ['log_dir_', 'inputs_shape_', 'outputs_shape_',
                 'data_placeholder_', 'labels_placeholder_',
                 'outputs_', 'metrics_', 'loss_',
                 'sess_', 'kwargs_',
                 'summary_', 'summary_writer_', 'projector_config_']

    def __init__(self, log_dir, inputs_shape, outputs_shape, inputs_type=tf.float32, outputs_type=tf.float32, reset_default_graph=True, metric_functions={}, **kwargs):
        print('Start initializing model...')

        # TensorBoard logging directory.
        self.log_dir_ = log_dir
        if tf.gfile.Exists(self.log_dir_):
            tf.gfile.DeleteRecursively(self.log_dir_)
        tf.gfile.MakeDirs(self.log_dir_)

        # Arguments.
        self.kwargs_ = kwargs

        # Reset default graph.
        if reset_default_graph:
            tf.reset_default_graph()

        # Input and Output layer shapes.
        self.inputs_shape_ = list(inputs_shape)
        self.outputs_shape_ = list(outputs_shape)

        # Generate placeholders for the data and labels.
        self.data_placeholder_ = tf.placeholder(inputs_type, shape=[None] + self.inputs_shape_, name='input_data')
        if self.labels_placeholder_ is None:
            self.labels_placeholder_ = tf.placeholder(outputs_type, shape=[None] + self.outputs_shape_, name='input_labels')

        # Build a Graph that computes predictions from the inference model.
        self.outputs_ = tf.identity(self.inference(self.data_placeholder_, **self.kwargs_), name='output_layer')

        # Loss function.
        self.loss_ = tf.identity(self.loss_function(self.outputs_, self.labels_placeholder_, **self.kwargs_), name='loss')

        # Evaluation options.
        self.metrics_ = {key : metric_functions[key](self.outputs_, self.labels_placeholder_) for key in metric_functions}
        self.metrics_['loss'] = self.loss_

        # Build the summary Tensor based on the TF collection of Summaries.
        for key in self.metrics_:
            tf.summary.scalar(key, self.metrics_[key])
        self.summary_ = tf.summary.merge_all()

        # Create a session for running Ops on the Graph.
        self.sess_ = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_ = tf.summary.FileWriter(self.log_dir_, self.sess_.graph)

        # Projector config object.
        self.projector_config_ = projector.ProjectorConfig()

        # And then after everything is built:

        # Run the Op to initialize the variables.
        self.sess_.run(tf.global_variables_initializer())

        print('Finish initializing model.')

    def inference(self, inputs, **kwargs):
        """Model inference."""
        raise Exception('Inference function should be overwritten!')
        return outputs

    def loss_function(self, outputs, labels_placeholder):
        """Loss function."""
        raise Exception('Loss function should be overwritten!')
        return loss

    def fit(self, train_set, iteration_count,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            epoch_count=None,
            val_set=None,
            summarizing_period=1,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000):
        """Train model."""

        if epoch_count is not None:
            assert epoch_count > 0, \
                'Epoch count should be greater than zero: epoch_count = %s' % epoch_count
            epoch_count = int(epoch_count)
        else:
            assert iteration_count is not None, \
                'Iteration count should be passed if epoch count is None: iteration_count = %s, epoch_count = %s' % (iteration_count, epoch_count)
        if iteration_count is not None:
            assert iteration_count > 0, \
                'Iteration count should be greater than zero: iteration_count = %s' % iteration_count
            iteration_count = int(iteration_count)
        if summarizing_period is not None:
            assert summarizing_period > 0, \
                'Summarizing period should be greater than zero: summarizing_period = %s' % summarizing_period
            summarizing_period = int(summarizing_period)
        if logging_period is not None:
            assert logging_period > 0, \
                'Logging period should be greater than zero: logging_period = %s' % logging_period
            logging_period = int(logging_period)
        if checkpoint_period is not None:
            assert checkpoint_period > 0, \
                'Checkpoint period should be greater than zero: checkpoint_period = %s' % checkpoint_period
            checkpoint_period = int(checkpoint_period)
        if evaluation_period is not None:
            assert evaluation_period > 0, \
                'Evaluation period should be greater than zero: evaluation_period = %s' % evaluation_period
            evaluation_period = int(evaluation_period)
        assert isinstance(train_set, TFDataset), \
            'Training set should be object of TFDataset type: type(train_set) = %s' % type(train_set)
        assert train_set.init_, \
            'Training set should be initialized: train_set.init_ = %s' % train_set.init_
        if val_set is not None:
            assert(isinstance(val_set, TFDataset)), \
                'Validation set should be object of TFDataset type: type(val_set) = %s' % type(val_set)
            assert val_set.init_, \
                'Validation set should be initialized: val_set.init_ = %s' % val_set.init_

        print('Start training iteration...')
        start_fit_time = time.time()

        # Checkpoint configuration.
        checkpoint_name = 'fit-checkpoint'
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = optimizer(learning_rate).minimize(self.loss_)

        # Run the Op to initialize the variables.
        self.sess_.run(tf.global_variables_initializer())

        # Get actual iteration and epoch count
        if epoch_count is not None:
            if iteration_count is not None:
                iteration_count = min(iteration_count, train_set.batch_count_ * epoch_count)
            else:
                iteration_count = train_set.batch_count_ * epoch_count
        else:
            epoch_count = iteration_count // train_set.batch_count_

        # Global iteration step and epoch number.
        iteration = 0
        epoch = 0

        # Start the training loop.
        iteration_times = []
        start_iteration_time = time.time()
        last_logging_iter = 0

        # Loop over all batches
        for batch in train_set.iterbatches(iteration_count):
            # Fill feed dict.
            feed_dict = {
                self.data_placeholder_: batch.data_,
                self.labels_placeholder_: batch.labels_,
            }

            # Run one step of the model training.
            self.sess_.run(train_op, feed_dict=feed_dict)

            # Save iteration time.
            iteration_times.append(time.time() - start_iteration_time)

            # Get current trained iteration and epoch.
            iteration += 1
            epoch = iteration // train_set.batch_count_

            # Write the summaries periodically.
            if iteration % summarizing_period == 0 or iteration == iteration_count:
                # Update the events file.
                summary_str = self.sess_.run(self.summary_, feed_dict=feed_dict)
                self.summary_writer_.add_summary(summary_str, iteration)

            # Print an overview periodically.
            if iteration % logging_period == 0 or iteration == iteration_count:
                # Calculate time of last period.
                duration = np.sum(iteration_times[-logging_period:])

                # Print logging info.
                metrics = self.evaluate(batch)
                metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                print('Iteration %d / %d (epoch %d / %d):   %s   [%.3f sec]' % (iteration, iteration_count, epoch, epoch_count, metrics_string, duration))

            # Save a checkpoint the model periodically.
            if checkpoint_period is not None and iteration % checkpoint_period == 0 or iteration == iteration_count:
                self.save(checkpoint_file, global_step=iteration)

            # Evaluate the model periodically.
            if evaluation_period is not None and iteration % evaluation_period == 0 or iteration == iteration_count:
                start_evaluation_time = time.time()
                metrics = self.evaluate(train_set)
                metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                duration = time.time() - start_evaluation_time
                print('Evaluation on full dataset:   [training   set]   %s   [%.3f sec]' % (metrics_string, duration))
                if val_set is not None:
                    start_evaluation_time = time.time()
                    metrics = self.evaluate(val_set)
                    metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                    duration = time.time() - start_evaluation_time
                    print('Evaluation on full dataset:   [validation set]   %s   [%.3f sec]' % (metrics_string, duration))

            start_iteration_time = time.time()

        self.summary_writer_.flush()
        total_time = time.time() - start_fit_time
        print('Finish training iteration (total time %.3f sec).\n' % total_time)

    def evaluate(self, dataset):
        """Evaluate model."""
        assert isinstance(dataset, TFDataset) or isinstance(dataset, TFBatch), \
            'Argument should be object of TFDataset or TFBatch type: type(dataset) = %s' % type(dataset)
        if isinstance(dataset, TFDataset):
            assert dataset.init_, \
                'Dataset should be initialized: dataset.init_ = %s' % dataset.init_
        if isinstance(dataset, TFBatch):
            assert hasattr(dataset, 'data_') and hasattr(dataset, 'labels_'), \
                'Batch should contain attributes \'data_\' and \'labels_\'.'
        result = {}
        if len(self.metrics_) > 0:
            metric_keys = list(self.metrics_.keys())
            metric_values = list(self.metrics_.values())
            estimates = self.sess_.run(metric_values, feed_dict={
                self.data_placeholder_: dataset.data_,
                self.labels_placeholder_: dataset.labels_,
            })
            for i in range(len(self.metrics_)):
                result[metric_keys[i]] = estimates[i]
        return result

    def save(self, filename, global_step=None):
        """Save checkpoint."""
        saver = tf.train.Saver(max_to_keep=None)
        saved_filename = saver.save(self.sess_, filename, global_step=global_step)
        print('Model saved to: %s' % saved_filename)

    def load(self, model_checkpoint_path=None):
        """Load checkpoint."""
        if model_checkpoint_path is None:
            model_checkpoint_path = tf.train.latest_checkpoint(self.log_dir_)
            assert model_checkpoint_path is not None, \
                'Checkpoint path automatically not found.'
        saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta', clear_devices=True)
        saver.restore(self.sess_, model_checkpoint_path)
        print('Model loaded from: %s' % model_checkpoint_path)

    @check_inputs_values
    def forward(self, inputs_values):
        """Forward propagation."""
        return self.sess_.run(self.outputs_, feed_dict={
            self.data_placeholder_: inputs_values,
        })

    @check_inputs_values
    def top_k(self, inputs_values, k):
        """Top k element."""
        return self.sess_.run(tf.nn.top_k(self.data_placeholder_, k=k), feed_dict={
            self.data_placeholder_: inputs_values,
        })

    def __str__(self):
        string = 'TFNeuralNetwork object:\n'
        for attr in self.__slots__:
            string += "%20s: %s\n" % (attr, getattr(self, attr))
        return string[:-1]

class TFClassifier(TFNeuralNetwork):

    """
    Classification model with Cross entropy loss function.
    """

    __slots__ = TFNeuralNetwork.__slots__ + ['probabilities_']

    def __init__(self, log_dir, inputs_shape, outputs_shape, metric_functions={}, **kwargs):
        if len(metric_functions) == 0:
            def accuracy(outputs, labels_placeholder):
                correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels_placeholder, 1))
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            metric_functions['accuracy'] = accuracy

        super().__init__(log_dir, inputs_shape, outputs_shape, inputs_type=inputs_type, outputs_type=outputs_type, metric_functions=metric_functions, **kwargs)
        self.probabilities_ = tf.nn.softmax(self.outputs_)
        
    def loss_function(self, outputs, labels_placeholder, **kwargs):
        """Cross entropy."""
        return tf.losses.softmax_cross_entropy(labels_placeholder, outputs) 

    @check_inputs_values
    def probabilities(self, inputs_values):
        """Get probabilites."""
        return self.sess_.run(self.probabilities_, feed_dict={
            self.data_placeholder_: inputs_values,
        })

    @check_inputs_values
    def classify(self, inputs_values):
        """Best prediction."""
        return self.sess_.run(tf.argmax(self.probabilities_, 1), feed_dict={
            self.data_placeholder_: inputs_values,
        })

class TFRegressor(TFNeuralNetwork):

    """
    Regression model with Mean squared loss function.
    """

    def loss_function(self, outputs, labels_placeholder):
        """Mean squared error."""
        return tf.losses.mean_squared_error(labels_placeholder, outputs)

class TFEmbedding(TFNeuralNetwork):

    """
    Embedding model with Triplet loss function.
    """

    @staticmethod
    def squared_distance(first_points, second_points):
        """Pairwise squared distances between 2 sets of points."""
        diff = tf.squared_difference(tf.expand_dims(first_points, 1), second_points)
        return tf.reduce_sum(diff, axis=2)

    def __init__(self, log_dir, inputs_shape, outputs_shape, inputs_type=tf.float32, outputs_type=tf.float32, reset_default_graph=True, metric_functions={}, **kwargs):

        if len(metric_functions) == 0:
            def max_accuracy(outputs, labels_placeholder):
                # Calculate distances
                embedding_pos, embedding_neg = tf.dynamic_partition(outputs, partitions=tf.reshape(labels_placeholder, [-1]), num_partitions=2)
                pos_dist = TFEmbedding.squared_distance(embedding_pos, embedding_pos)
                neg_dist = TFEmbedding.squared_distance(embedding_pos, embedding_neg)
                tf.summary.histogram('pos_dist', pos_dist)
                tf.summary.histogram('neg_dist', neg_dist)

                def triplet_accuracy(pos_dist, neg_dist):
                    """Triplet accuracy function for binary classification to positives and negatives."""
                    def accuracy(threshold):
                        correct_count = tf.count_nonzero(pos_dist < threshold) + tf.count_nonzero(neg_dist >= threshold)
                        total_count = tf.shape(pos_dist)[0] * tf.shape(pos_dist)[1] + tf.shape(neg_dist)[0] * tf.shape(neg_dist)[1]
                        return tf.cast(correct_count, dtype=tf.float32) / tf.cast(total_count, dtype=tf.float32)
                    return accuracy

                # Get all possible threshold values
                total_dist = tf.reshape(tf.concat([pos_dist, neg_dist], 1), [-1])
                thresholds = tf.unique(total_dist)[0]

                # Calculate accuracy
                accuracies = tf.map_fn(triplet_accuracy(pos_dist, neg_dist), thresholds)
                return tf.reduce_max(accuracies)

            metric_functions['max_accuracy'] = max_accuracy

        # Reset default graph.
        if reset_default_graph:
            tf.reset_default_graph()

        self.labels_placeholder_ = tf.placeholder(tf.int32, shape=[None], name='input_labels')
        super().__init__(log_dir, inputs_shape, outputs_shape, inputs_type=inputs_type, outputs_type=outputs_type, reset_default_graph=False, metric_functions=metric_functions, **kwargs)

    def loss_function(self, outputs, labels_placeholder, **kwargs):
        """Compute the triplet loss by mini-batch of triplet embeddings."""
        assert 'margin' in kwargs, \
            'Argument \'margin\' should be passed: kwargs = %s' % kwargs
        assert 'exclude_hard' in kwargs, \
            'Argument \'exclude_hard\' should be passed: kwargs = %s' % kwargs
        margin = kwargs['margin']
        exclude_hard = kwargs['exclude_hard']

        def triplet_loss(margin, exclude_hard):
            """Triplet loss function for a given anchor."""
            def loss(pos_neg_dist):
                pos_dist, neg_dist = pos_neg_dist
                raw_loss = tf.expand_dims(pos_dist, -1) - neg_dist + margin
                mask = raw_loss > 0
                if exclude_hard:
                    mask = tf.logical_and(mask, raw_loss < margin)
                valid_loss = raw_loss * tf.cast(mask, dtype=tf.float32)
                return valid_loss
            return loss

        # Calculate distances
        embedding_pos, embedding_neg = tf.dynamic_partition(outputs, partitions=self.labels_placeholder_, num_partitions=2)
        pos_dist = TFEmbedding.squared_distance(embedding_pos, embedding_pos)
        neg_dist = TFEmbedding.squared_distance(embedding_pos, embedding_neg)

        # Calculate losses
        losses = tf.map_fn(triplet_loss(margin, exclude_hard), (pos_dist, neg_dist), dtype=tf.float32)
        return tf.reduce_mean(losses)

    def fit(self, train_set, iteration_count,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            epoch_count=None,
            val_set=None,
            summarizing_period=1,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000):
        """Train model."""
        
        assert isinstance(train_set, TFTripletset), \
            'Training set should be object of TFTripletset type: type(train_set) = %s' % type(train_set)
        if val_set is not None:
            assert isinstance(val_set, TFTripletset), \
                'Validation set should be object of TFTripletset type: type(val_set) = %s' % type(val_set)

        super().fit(train_set=train_set, iteration_count=iteration_count,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    epoch_count=epoch_count,
                    val_set=val_set,
                    summarizing_period=summarizing_period,
                    logging_period=logging_period,
                    checkpoint_period=checkpoint_period,
                    evaluation_period=evaluation_period)

    def evaluate(self, dataset):
        """Evaluate model."""
        if isinstance(dataset, TFDataset):
            warnings.warn('Evaluation function is not implemented for TFDataset!', Warning)
            return {}
        else:
            return super().evaluate(dataset)

    @check_inputs_values
    def visualize(self, inputs_values, var_name, labels=None):
        """Visualize embeddings in TensorBoard."""
        if labels is not None:
            assert len(inputs_values) == len(labels), \
                'Inputs values and labels should be the same lengths: len(inputs_values) = %s, len(labels) = %s' % (len(inputs_values), len(labels))

        # Get visualization embeddings
        vis_embeddings = self.forward(inputs_values)
        if labels is not None:
            vis_labels = labels.flatten()
        vis_name = tf.get_default_graph().unique_name(var_name, mark_as_used=False)
        
        # Input set for Embedded TensorBoard visualization
        vis_var = tf.Variable(tf.stack(vis_embeddings, axis=0), trainable=False, name=vis_name)
        self.sess_.run(tf.variables_initializer([vis_var]))

        # Add embedding tensorboard visualization
        embed = self.projector_config_.embeddings.add()
        embed.tensor_name = vis_name + ':0'
        if labels is not None:
            embed.metadata_path = os.path.join(self.log_dir_, embed.tensor_name + '_metadata.tsv')
        projector.visualize_embeddings(self.summary_writer_, self.projector_config_)

        # Checkpoint configuration
        checkpoint_name = 'vis-checkpoint'
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)

        # Save checkpoint
        self.save(checkpoint_file)

        # Write labels info
        if labels is not None:
            with open(embed.metadata_path, 'w') as f:
                is_first = True
                for label in vis_labels:
                    if is_first:
                        f.write(str(label))
                        is_first = False
                    else:
                        f.write('\n' + str(label))

        # Print status info
        print('For watching in TensorBoard run command:\ntensorboard --logdir "%s"' % self.log_dir_)
