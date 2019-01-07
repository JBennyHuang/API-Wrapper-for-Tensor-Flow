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

        accuracy_op = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(y_train, 1)), tf.float32))

        loss_op = loss(predictions, y_train)

        global_step = tf.train.create_global_step()

        optimizer_op = optimizer.minimize(loss_op, global_step=global_step)

        self.session.run(tf.global_variables_initializer())

        if tf.train.latest_checkpoint(path) is not None:
            self.restore(path)
            print('restoring from checkpoint')

        for i in range(num_step):

            g, a, l, _ = self.session.run(
                [global_step, accuracy_op, loss_op, optimizer_op])

            print(f'step: {g} - accuracy: {a} - loss: {l}')

            if l < self.best_loss:
                self.best_loss = l
                self.save(path)
                print(f'saving checkpoint on step: {g}')

    def predict(self, input_fn, architecture, input_shape, path):

        predictions = architecture(
            tf.placeholder(tf.float32, shape=input_shape, name='input'), False)

        self.session.run(tf.global_variables_initializer())

        if tf.train.latest_checkpoint(path) is not None:
            self.restore(path)
            print('restoring from checkpoint')

        for x in input_fn():
            yield self.session.run(predictions, feed_dict={'input:0': x})
