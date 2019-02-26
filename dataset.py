import tensorflow as tf
import numpy as np


class Dataset:

    def __init__(self, x_train_paths, y_train_paths, load_paths, data_parser, data_preprocessor, batch_size):
        self.session = tf.Session()

        x_train, y_train = load_paths(x_train_paths, y_train_paths)

        def data_parser_fn(x_train, y_train):
            output = tf.py_func(data_parser, [x_train, y_train], [
                tf.float32, tf.float32])
            return output

        def data_preprocessor_fn(x_train, y_train):
            output = tf.py_func(data_preprocessor, [
                x_train, y_train], [tf.float32, tf.float32])
            return output

        print(x_train.shape)
        print(y_train.shape)

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(len(x_train))
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.map(data_parser_fn, 4)
        dataset = dataset.map(data_preprocessor_fn, 4)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)

        self.dataset_iterator = dataset.make_one_shot_iterator()

    def next_batch(self):
        return self.dataset_iterator.get_next()
