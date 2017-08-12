import tensorflow as tf
import numpy as np
import warnings
import math
import os
import time
import pickle
from tensorflow.contrib.tensorboard.plugins import projector

class TFDataset:
    __slots__ = ['init_', 'size_', 'data_', 'labels_', 'data_shape_', 'labels_shape_', 'batch_size_', 'batch_count_', 'batch_num_']

    batch_num_ = 0
    init_ = False

    def __init__(self, data=None, labels=None, batch_size=1):
        if data is not None and labels is not None:
            self.set(data, labels, batch_size)
        else:
            for attr in self.__slots__:
                if attr != 'init_' and attr != 'batch_num_':
                    setattr(self, attr, None)
        
    def set(self, data, labels, batch_size=1):
        """Set values."""
        assert(len(data) == len(labels))
        assert(batch_size > 0)
        assert(batch_size <=  len(data))
        
        data = np.array(data)
        labels = np.array(labels)
        
        self.size_ = len(data)
        if data.ndim == 1:
            self.data_ = np.reshape(data, (self.size_, 1))
        else:
            self.data_ = data
        if labels.ndim == 1:
            self.labels_ = np.reshape(labels, (self.size_, 1))
        else:
            self.labels_ = labels
        self.data_shape_ = list(self.data_.shape)
        self.labels_shape_ = list(self.labels_.shape)
        self.data_ndim_ = len(self.data_shape_) - 1
        self.labels_ndim_ = len(self.labels_shape_) - 1
        self.batch_size_ = batch_size
        self.batch_count_ = int(self.size_ / self.batch_size_)
        self.init_ = True
        
    def shuffle(self):
        """Random shuffling dataset."""
        assert(self.init_)
        indexes = np.arange(self.size_)
        np.random.shuffle(indexes)
        self.data_ = self.data_[indexes]
        self.labels_ = self.labels_[indexes]
    
    def next_batch(self):
        """Get next batch."""
        assert(self.init_)
        first = (self.batch_num_ * self.batch_size_) % self.size_
        last = first + self.batch_size_
        batch_data = None
        batch_labels = None
        if (last <= self.size_):
            batch_data = self.data_[first:last]
            batch_labels = self.labels_[first:last]
        else:
            batch_data = np.append(self.data_[first:], self.data_[:last - self.size_], axis=0)
            batch_labels = np.append(self.labels_[first:], self.labels_[:last - self.size_], axis=0)
        self.batch_num_ += 1
        return batch_data, batch_labels
    
    def batches(self):
        """Get generator by batches."""
        assert(self.init_)
        i = 0
        while i + self.batch_size_ <= self.size_:
            batch_data = self.data_[i : i + self.batch_size_]
            batch_labels = self.labels_[i : i + self.batch_size_]
            indicators = np.ones(self.batch_size_).astype(bool)
            yield batch_data, batch_labels, indicators
            i += self.batch_size_
        residue = dataset.size_ % dataset.batch_size_
        if residue != 0:
            batch_data = self.data_[-self.batch_size_ : ]
            batch_labels = self.labels_[-self.batch_size_ : ]
            indicators = np.append(np.zeros(self.batch_size_ - residue), np.ones(residue)).astype(bool)
            yield batch_data, batch_labels, indicators
    
    def split(self, train_size, val_size, test_size, shuffle=True):
        """Split dataset to train, validation and test set."""
        assert(self.init_)
        assert(train_size >= 0)
        assert(val_size >= 0)
        assert(test_size >= 0)
        assert(train_size + val_size + test_size == self.size_)
        if shuffle:
            self.shuffle()
        if train_size > 0:
            train_set = TFDataset(self.data_[:train_size], self.labels_[:train_size], self.batch_size_)
        else:
            train_set = None
        if val_size > 0:
            val_set = TFDataset(self.data_[train_size:train_size + val_size], self.labels_[train_size:train_size + val_size], self.batch_size_)
        else:
            val_set = None
        if test_size > 0:
            test_set = TFDataset(self.data_[-test_size:], self.labels_[-test_size:], self.batch_size_)
        else:
            test_set = None
        return train_set, val_set, test_set
    
    def load(self, filename):
        """Load dataset from file."""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            for attr in self.__slots__:
                setattr(self, attr, getattr(obj, attr))

    def save(self, filename):
        """Save dataset to file."""
        assert(self.init_)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
    def __str__(self):
        string = "TFDataset object:\n"
        for attr in self.__slots__:
            string += "%20s: %s\n" % (attr, getattr(self, attr))
        return string

class TFNeuralNetwork(object):
    __slots__ = ['inputs_shape_', 'outputs_shape_', 'data_placeholder_', 'labels_placeholder_', 'outputs_', 'metrics_', 'loss_',
                 'log_dir_', 'sess_', 'kwargs_', 'summary_', 'summary_writer_']

    def __init__(self, log_dir, inputs_shape, outputs_shape, metric_functions={}, kwargs={}):
        print('Start initializing model...')
        
        # TensorBoard logging directory.
        self.log_dir_ = log_dir
        if tf.gfile.Exists(self.log_dir_):
            tf.gfile.DeleteRecursively(self.log_dir_)
        tf.gfile.MakeDirs(self.log_dir_)
        
        # Input and Output layer shapes.
        self.inputs_shape_ = list(inputs_shape)
        self.outputs_shape_ = list(outputs_shape)

        # Arguments.
        self.kwargs_ = kwargs

        # Generate placeholders for the data and labels.
        self.data_placeholder_ = tf.placeholder(tf.float32, shape=[None] + self.inputs_shape_, name="data")
        self.labels_placeholder_ = tf.placeholder(tf.int32, shape=[None] + self.outputs_shape_, name="labels")

        # Build a Graph that computes predictions from the inference model.
        self.outputs_ = self.inference(self.data_placeholder_, self.kwargs_)

        # Loss function.
        self.loss_ = self.loss_function(self.outputs_, self.labels_placeholder_)

        # Evaluation options.
        self.metrics_ = {key : metric_functions[key](self.outputs_, self.labels_placeholder_) for key in metric_functions}
        self.metrics_['loss'] = self.loss_
        
        # Build the summary Tensor based on the TF collection of Summaries.
        for key in self.metrics_:
            tf.summary.scalar(key, self.metrics_[key])
        self.summary_ = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        self.sess_ = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_ = tf.summary.FileWriter(self.log_dir_, self.sess_.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        self.sess_.run(init)
        
        print('Finish initializing model.')
        
    def inference(self, inputs, kwargs={}):
        """Model inference."""
        raise Exception('Inference function should be overwritten!')
        return outputs

    def loss_function(self, outputs, labels_placeholder):
        """Loss function."""
        raise Exception('Loss function should be overwritten!')
        return loss

    def fit(self, train_set, epoch_count,
        optimizer=tf.train.RMSPropOptimizer,
        learning_rate=0.001,
        val_set=None,
        checkpoint_period=1000,
        summarizing_period=1):
        """Train model."""

        assert(epoch_count > 0)
        assert(isinstance(train_set, TFDataset))
        assert(val_set is None or isinstance(val_set, TFDataset))
        assert(val_set is None or train_set.data_ndim_ == val_set.data_ndim_)
        assert(val_set is None or train_set.labels_ndim_ == val_set.labels_ndim_)
        assert(val_set is None or np.all(np.array(train_set.labels_shape_[1:]) == np.array(val_set.labels_shape_[1:])))
        assert(val_set is None or np.all(np.array(train_set.data_shape_[1:]) == np.array(val_set.data_shape_[1:])))

        # Checkpoint configuration.
        checkpoint_name = "fit-checkpoint"
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)
        
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = optimizer(learning_rate).minimize(self.loss_)
        
        # Add the variable initializer Op.
        self.sess_.run(tf.global_variables_initializer())
        
        print('Start training iteration...')
        
        # Start the training loop.
        for epoch in range(epoch_count):
            start_time = time.time()

            # Loop over all batches
            for i in range(train_set.batch_count_):
                # Get next batch.
                batch_data, batch_labels = train_set.next_batch()

                # Fill feed dict.
                feed_dict = {
                  self.data_placeholder_: batch_data,
                  self.labels_placeholder_: batch_labels,
                }
                
                # Run one step of the model training.
                self.sess_.run(train_op, feed_dict = feed_dict)
                
                # Update the events file.
                summary_str = self.sess_.run(self.summary_, feed_dict = feed_dict)
                self.summary_writer_.add_summary(summary_str, epoch)
                
            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if (epoch + 1) % summarizing_period == 0:
                if val_set is not None:
                    metrics = self.evaluate(val_set)
                else:
                    metrics = self.evaluate(train_set)
                metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                print('Epoch %d/%d:   %s   (%.3f sec)' % (epoch + 1, epoch_count, metrics_string, duration))
                self.summary_writer_.flush()
                
            # Save a checkpoint and evaluate the model periodically.
            if (epoch + 1) % checkpoint_period == 0 or (epoch + 1) == epoch_count:
                self.save(checkpoint_file, global_step=epoch)
        
        print('Finish training iteration.')
            
    def evaluate(self, dataset):
        """Evaluate model."""
        assert(isinstance(dataset, TFDataset))
        result = {}
        if len(self.metrics_) > 0:
            metric_names = []
            metric_values = []
            for key in self.metrics_:
                metric_names.append(key)
                metric_values.append(self.metrics_[key])
            estimates = self.sess_.run(metric_values, feed_dict = {
              self.data_placeholder_: dataset.data_,
              self.labels_placeholder_: dataset.labels_,
            })
            for i in xrange(len(self.metrics_)):
                result[metric_names[i]] = estimates[i]
        return result

    def save(self, filename, global_step=None):
        """Save checkpoint."""
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        saver.save(self.sess_, filename, global_step=global_step)
        print('Model saved to: %s' % filename)

    def forward(self, inputs_values):
        """Forward propagation."""
        assert(np.all(np.array(inputs_values.shape[1:]) == np.array(self.inputs_shape_)))
        return self.sess_.run(self.outputs_, feed_dict = {
          self.data_placeholder_: inputs_values,
        })

    def top_k(self, inputs_values, k):
        """Top k element."""
        return self.sess_.run(tf.nn.top_k(self.data_placeholder_, k=k), feed_dict = {
          self.data_placeholder_: inputs_values,
        })

    def __str__(self):
        string = "TFNeuralNetwork object:\n"
        for attr in self.__slots__:
            string += "%20s: %s\n" % (attr, getattr(self, attr))
        return string

class TFClassifier(TFNeuralNetwork):
    __slots__ = TFNeuralNetwork.__slots__ + ['probabilities_']

    def __init__(self, log_dir, inputs_shape, outputs_shape, metric_functions={}, kwargs={}):
        if len(metric_functions) == 0:
            def accuracy(outputs, labels_placeholder):
                correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels_placeholder, 1))
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            metric_functions["accuracy"] = accuracy

        super(TFClassifier, self).__init__(log_dir, inputs_shape, outputs_shape, metric_functions, kwargs)
        self.probabilities_ = tf.nn.softmax(self.outputs_)
        
    def loss_function(self, outputs, labels_placeholder):
        """Cross entropy."""
        return tf.losses.softmax_cross_entropy(labels_placeholder, outputs) 

    def probabilities(self, inputs_values):
        """Get probabilites."""
        return self.sess_.run(self.probabilities_, feed_dict = {
          self.data_placeholder_: inputs_values,
        })

    def classify(self, inputs_values):
        """Best prediction."""
        return self.sess_.run(tf.argmax(self.probabilities_, 1), feed_dict = {
          self.data_placeholder_: inputs_values,
        })

class TFRegressor(TFNeuralNetwork):

    def __init__(self, log_dir, inputs_shape, outputs_shape, metric_functions={}, kwargs={}):
        super(TFRegressor, self).__init__(log_dir, inputs_shape, outputs_shape, metric_functions, kwargs)
        
    def loss_function(self, outputs, labels_placeholder):
        """Mean squared error."""
        return tf.losses.mean_squared_error(labels_placeholder, outputs)

class TFEmbeddingTripletLoss(TFNeuralNetwork):

    def __init__(self, log_dir, inputs_shape, outputs_shape, metric_functions={}, kwargs={}):
        super(TFEmbeddingTripletLoss, self).__init__(log_dir, inputs_shape, outputs_shape, metric_functions, kwargs)
        self.labels_placeholder_ = None

    @staticmethod
    def euclidean_distance_squared(x, y):
        """
        Compute square of euclidean distance between two tensorflow variables
        """
        return tf.reduce_sum(tf.square(tf.subtract(x, y)), 1, keep_dims=True)

    @staticmethod
    def triplet_loss(anchor_emb, positive_emb, negative_emb, margin):
        """
        Compute the triplet loss by anchor, positive and negative embeddings as:
        L = [ || f_a - f_p ||^2 - || f_a - f_n ||^2 + m ]+
        """
        with tf.name_scope("triplet_loss"):
            pos_dist = TFEmbeddingTripletLoss.euclidean_distance_squared(anchor_emb, positive_emb)
            neg_dist = TFEmbeddingTripletLoss.euclidean_distance_squared(anchor_emb, negative_emb)
            loss = tf.maximum(0., pos_dist - neg_dist + margin)
            return tf.reduce_mean(loss)

    @staticmethod
    def sample_batch(data, labels, class_count_per_batch, data_count_per_class):
        labels = labels.flatten()
        old_indexes = set()
        batch_data = []
        batch_labels = []

        # Get random classes.
        unique_labels = np.unique(labels)
        np.random.shuffle(unique_labels)
        for i in xrange(class_count_per_batch):
            rand_label = unique_labels[i]

            # Get new random data in current class.
            for _ in xrange(data_count_per_class):
                rand_ind = np.random.randint(len(data))
                while rand_ind in old_indexes or labels[rand_ind] != rand_label:
                    rand_ind = np.random.randint(len(data))

                # Save selected data and class pair.
                batch_data.append(data[rand_ind])
                batch_labels.append(labels[rand_ind])
                old_indexes.add(rand_ind)
        return np.array(batch_data), np.array(batch_labels)

    @staticmethod
    def stack_triplets(data, embeddings, labels, margin):
        labels = labels.flatten()
        triplets_data = []
        triplets_labels = []

        # Get all possible anchors.
        for anchor_ind in xrange(len(labels) - 1):
            # Check if next is the same label.
            if labels[anchor_ind + 1] != labels[anchor_ind]:
                continue
                
            # Calculate squared distances to negative embeddings.
            neg_dists = np.square(np.linalg.norm(embeddings[anchor_ind] - embeddings, axis=1))
            neg_dists[labels == labels[anchor_ind]] = np.NaN
                
            # Get all possible positives.
            positive_ind = anchor_ind + 1
            while positive_ind < len(labels) and labels[positive_ind] == labels[anchor_ind]:
                # Calculate squared distances to positive.
                pos_dist = np.square(np.linalg.norm(embeddings[anchor_ind] - embeddings[positive_ind]))
                
                # Get indexes of semi-hard negatives.
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<margin, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(pos_dist - neg_dists + margin > 0)[0] # VGG Face selection
                
                # Check if semi-hard negatives exist.
                if len(all_neg) > 0:
                    # Get random negative.
                    rand_ind = np.random.randint(len(all_neg))
                    negative_ind = all_neg[rand_ind]
                    
                    # Add triplet.
                    triplets_data.append((data[anchor_ind], data[positive_ind], data[negative_ind]))
                    triplets_labels.append((labels[anchor_ind], labels[positive_ind], labels[negative_ind]))
                
                # Go to next positive.
                positive_ind += 1
                
        # Stack triplets to rows of features and return.
        return np.array(triplets_data).reshape((-1, data.shape[-1])), np.array(triplets_labels).reshape((-1, 1))

    def loss_function(self, outputs, labels_placeholder):
        """
        Compute the triplet loss by mini-batch of stacked triplet embeddings:
        (3i+0)-th correspond to anchor
        (3i+1)-th correspond to positive
        (3i+2)-th correspond to negative
        """
        assert('margin' in self.kwargs_)
        reshape_emb = tf.reshape(outputs, (-1, 3, int(outputs.shape[-1])))
        anchor_emb, positive_emb, negative_emb = tf.unstack(reshape_emb, axis=1)
        return TFEmbeddingTripletLoss.triplet_loss(anchor_emb, positive_emb, negative_emb, self.kwargs_['margin'])

    def fit(self, train_set, iter_count, class_count_per_batch, data_count_per_class,
        optimizer=tf.train.RMSPropOptimizer,
        learning_rate=0.001,
        val_set=None,
        val_data_count_per_class=None,
        checkpoint_period=10000,
        summarizing_period=100):
        """Train model."""

        assert(iter_count > 0)
        assert(isinstance(train_set, TFDataset))
        assert(train_set.labels_ndim_ == 1)
        if val_set is not None:
            assert(isinstance(val_set, TFDataset))
            assert(train_set.data_ndim_ == val_set.data_ndim_)
            assert(train_set.labels_ndim_ == val_set.labels_ndim_)
            assert(np.all(np.array(train_set.labels_shape_[1:]) == np.array(val_set.labels_shape_[1:])))
            assert(np.all(np.array(train_set.data_shape_[1:]) == np.array(val_set.data_shape_[1:])))
            assert(val_data_count_per_class is not None) 

        # Checkpoint configuration.
        checkpoint_name = "fit-checkpoint"
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)
        
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = optimizer(learning_rate).minimize(self.loss_)
        
        # Add the variable initializer Op.
        self.sess_.run(tf.global_variables_initializer())
        
        # Prepare validation triplets.
        if val_set is not None:
            unique_labels_count = len(np.unique(val_set.labels_))
            val_data, val_labels = TFEmbeddingTripletLoss.sample_batch(val_set.data_, val_set.labels_, unique_labels_count, val_data_count_per_class)
            val_embeddings = self.forward(val_data)
            val_triplets_data, val_triplets_labels = TFEmbeddingTripletLoss.stack_triplets(val_data, val_embeddings, val_labels, self.kwargs_['margin'])

        print('Start training iteration...')
        
        # Start the training loop.
        for iteration in range(iter_count):
            start_time = time.time()

            # Get next batch.
            batch_data, batch_labels = TFEmbeddingTripletLoss.sample_batch(train_set.data_, train_set.labels_, class_count_per_batch, data_count_per_class)

            # Calculate batch embeddings.
            batch_embeddings = self.forward(batch_data)

            # Get triplets from batch.
            batch_triplets_data, batch_triplets_labels = TFEmbeddingTripletLoss.stack_triplets(batch_data, batch_embeddings, batch_labels, self.kwargs_['margin'])

            # Fill feed dict.
            feed_dict = {
                self.data_placeholder_: batch_triplets_data,
            }

            # Run one step of the model training.
            self.sess_.run(train_op, feed_dict=feed_dict)

            # Update the events file.
            summary_str = self.sess_.run(self.summary_, feed_dict=feed_dict)
            self.summary_writer_.add_summary(summary_str, iteration)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if (iteration + 1) % summarizing_period == 0:
                batch_loss = self.loss_.eval(session=self.sess_, feed_dict={
                    self.data_placeholder_: batch_triplets_data
                })
                if val_set is not None:
                    val_loss = self.loss_.eval(session=self.sess_, feed_dict={
                        self.data_placeholder_: val_triplets_data
                    })
                    print('Iteration %d/%d:   batch_loss = %.6f   val_loss = %.6f' % (iteration + 1, iter_count, batch_loss, val_loss))
                else:
                    print('Iteration %d/%d:   batch_loss = %.6f' % (iteration + 1, iter_count, batch_loss))
                self.summary_writer_.flush()
                
            # Save a checkpoint and evaluate the model periodically.
            if (iteration + 1) % checkpoint_period == 0 or (iteration + 1) == iter_count:
                self.save(checkpoint_file, global_step=iteration)
        
        print('Finish training iteration.')

    def evaluate(self, dataset):
        """Evaluate model."""
        raise Exception('Evaluate function not implemented!')
        return result

    def visualize(self, dataset, embedding_count):
        """Visualize embeddings in TensorBoard."""
        assert(isinstance(dataset, TFDataset))

        # Get visualization embeddings
        vis_embeddings = self.forward(dataset.data_[:embedding_count])
        vis_labels = dataset.labels_[:embedding_count].flatten()

        # Input set for Embedded TensorBoard visualization
        embedding_var = tf.Variable(tf.stack(vis_embeddings, axis=0), trainable=False, name='embedding_var')
        self.sess_.run(tf.variables_initializer([embedding_var]))

        # Add embedding tensorboard visualization. Need tensorflow version
        # >= 0.12.0RC0
        config = projector.ProjectorConfig()
        embed= config.embeddings.add()
        embed.tensor_name = 'embedding_var:0'
        embed.metadata_path = os.path.join(self.log_dir_, 'metadata.tsv')
        projector.visualize_embeddings(self.summary_writer_, config)

        # Checkpoint configuration.
        checkpoint_name = "emb-checkpoint"
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)

        # Save checkpoint
        self.save(checkpoint_file)

        # Write labels info
        with open(embed.metadata_path, 'w') as f:
            is_first = True
            for label in vis_labels:
                if is_first:
                    f.write(str(label))
                    is_first = False
                else:
                    f.write('\n' + str(label))

        # Print status info.
        print("For watching embedding in TensorBoard run command:")
        print("tensorboard --logdir '%s'" % self.log_dir_)
