import tensorflow as tf
import numpy as np

from glob import glob


class Dataset:
    def load_filenames(self, dirs):
        return np.column_stack(sorted(glob('{}/*.png'.format(d))) for d in dirs)

    def load_image(self, filename, channels):
        image = tf.read_file(filename)
        image = tf.image.decode_png(image, channels=channels, dtype=tf.uint16)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

    def reshape_tensor(self, tensor):
        tensor = tf.unstack(tensor)
        tensor = tf.concat(tensor, 2)
        return tensor

    def parse_fn(self, x_train, y_train):
        x_train = tf.map_fn(lambda e: self.load_image(
            e, self.x_channels), x_train, dtype=tf.float32)
        y_train = tf.map_fn(lambda e: self.load_image(
            e, self.y_channels), y_train, dtype=tf.float32)
        return self.reshape_tensor(x_train), self.reshape_tensor(y_train)

    def preproc_fn(self, x_train, y_train):
        return x_train, y_train

    def __init__(self, x_train_dirs, y_train_dirs, x_channels, y_channels, batch_size, num_parallel_calls=2, prefetch_size_batch=1):
        self.x_channels = x_channels
        self.y_channels = y_channels

        x_train, y_train = self.load_filenames(
            x_train_dirs), self.load_filenames(y_train_dirs)

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(len(x_train))
        dataset = dataset.map(self.parse_fn, num_parallel_calls)
        dataset = dataset.map(self.preproc_fn, num_parallel_calls)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size_batch)

        self.dataset_iterator = dataset.make_one_shot_iterator()

    def next_batch(self):
        return self.dataset_iterator.get_next()
