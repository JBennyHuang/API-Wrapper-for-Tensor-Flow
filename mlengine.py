import tensorflow as tf

import math

from dataset import Dataset

class Model:
    def __init__(self):
        self.session = tf.Session()
        self.best_loss = math.inf

    def save(self, path):
        tf.train.Saver().save(self.session, f'{path}/model.ckpt')
        f = open(f'{path}/loss.txt', 'a')
        f.write(f'{str(self.best_loss)}\n')
        f.close()

    def restore(self, path):
        tf.train.Saver().restore(self.session, f'{path}/model.ckpt')
        f = open(f'{path}/loss.txt', 'r')
        self.best_loss = float(f.readlines()[-1])
        f.close()

    def train(self, input_fn, architecture, loss, optimizer, num_step, path):

        x_train, y_train = input_fn()

        predictions = architecture(x_train, True)

        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(y_train, 1)), tf.float32))

        loss_op = loss(predictions, y_train)

        global_step = tf.train.create_global_step()

        optimizer_op = optimizer.minimize(loss_op, global_step=global_step)

        if tf.train.latest_checkpoint(path) is not None:
            self.restore(path)
            print('restoring from checkpoint')

        self.session.run(tf.global_variables_initializer())

        for i in range(num_step):

            g, a, l, _ = self.session.run([global_step, accuracy_op, loss_op, optimizer_op])

            print(f'step: {g} - accuracy: {a} - loss: {l}')
            if l < self.best_loss:
                self.best_loss = l
                self.save(path)
                print(f'saving checkpoint on step: {g}')



def can16(x, is_training):
    y = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=1)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=2)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=4)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=8)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=16)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=32)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=64)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=128)
    y = tf.layers.conv2d(y, 16, 3, padding='same', activation=tf.nn.relu, dilation_rate=1)
    y = tf.layers.conv2d(y, 1, 3, padding='same')

    return y

def rmse(predictions, y_train):
    return tf.sqrt(tf.reduce_mean(tf.square(predictions - y_train)))

input_fn = Dataset(['./train_x/p1', './train_x/p2'], ['./train_y'], 5).next_batch

m = Model()

m.train(input_fn, can16, rmse, tf.train.AdamOptimizer(0.0001), 100, './test')