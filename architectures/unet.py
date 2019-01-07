import tensorflow as tf


def unet(x, is_training):
    l1 = tf.layers.conv2d(x, 32, (3, 3), padding='same',
                          activation=tf.nn.relu)
    l2 = tf.layers.conv2d(l1, 32, (3, 3), padding='same',
                          activation=tf.nn.relu)
    l3 = tf.layers.max_pooling2d(l2, (2, 2), (2, 2), padding='same')

    l4 = tf.layers.conv2d(l3, 32, (3, 3), padding='same',
                          activation=tf.nn.relu)
    l5 = tf.layers.conv2d(l4, 32, (3, 3), padding='same',
                          activation=tf.nn.relu)
    l6 = tf.layers.max_pooling2d(l5, (2, 2), (2, 2), padding='same')

    l7 = tf.layers.conv2d(l6, 32, (3, 3), padding='same',
                          activation=tf.nn.relu)
    l8 = tf.layers.conv2d(l7, 32, (3, 3), padding='same',
                          activation=tf.nn.relu)
    l9 = tf.layers.max_pooling2d(l8, (2, 2), (2, 2), padding='same')

    l10 = tf.layers.conv2d(l9, 64, (3, 3), padding='same',
                           activation=tf.nn.relu)
    l11 = tf.layers.conv2d(
        l10, 64, (3, 3), padding='same', activation=tf.nn.relu)
    l12 = tf.layers.max_pooling2d(l11, (2, 2), (2, 2), padding='same')

    l13 = tf.layers.conv2d(
        l12, 128, (3, 3), padding='same', activation=tf.nn.relu)
    l14 = tf.layers.conv2d(
        l13, 128, (3, 3), padding='same', activation=tf.nn.relu)
    l15 = tf.layers.max_pooling2d(l14, (2, 2), (2, 2), padding='same')

    l16 = tf.layers.conv2d(
        l15, 256, (3, 3), padding='same', activation=tf.nn.relu)
    l17 = tf.layers.conv2d(
        l16, 256, (3, 3), padding='same', activation=tf.nn.relu)
    l18 = tf.layers.max_pooling2d(l17, (2, 2), (2, 2), padding='same')

    l19 = tf.layers.conv2d(
        l18, 512, (3, 3), padding='same', activation=tf.nn.relu)
    l20 = tf.layers.conv2d(
        l19, 512, (3, 3), padding='same', activation=tf.nn.relu)

    l20_shape = tf.shape(l20)
    l20_resized = tf.image.resize_bilinear(
        l20, size=[2*l20_shape[1], 2*l20_shape[2]], align_corners=True)
    l21 = tf.concat([l20_resized, l17], axis=3)
    l22 = tf.layers.conv2d(
        l21, 256, (3, 3), padding='same', activation=tf.nn.relu)
    l23 = tf.layers.conv2d(
        l22, 256, (3, 3), padding='same', activation=tf.nn.relu)
    l24 = tf.layers.conv2d(
        l23, 256, (3, 3), padding='same', activation=tf.nn.relu)

    l24_shape = tf.shape(l24)
    l24_resized = tf.image.resize_bilinear(
        l24, size=[2*l24_shape[1], 2*l24_shape[2]], align_corners=True)
    l25 = tf.concat([l24_resized, l14], axis=3)
    l26 = tf.layers.conv2d(
        l25, 128, (3, 3), padding='same', activation=tf.nn.relu)
    l27 = tf.layers.conv2d(
        l26, 128, (3, 3), padding='same', activation=tf.nn.relu)
    l28 = tf.layers.conv2d(
        l27, 128, (3, 3), padding='same', activation=tf.nn.relu)

    l28_shape = tf.shape(l28)
    l28_resized = tf.image.resize_bilinear(
        l28, size=[2*l28_shape[1], 2*l28_shape[2]], align_corners=True)
    l29 = tf.concat([l28_resized, l11], axis=3)
    l30 = tf.layers.conv2d(
        l29, 64, (3, 3), padding='same', activation=tf.nn.relu)
    l31 = tf.layers.conv2d(
        l30, 64, (3, 3), padding='same', activation=tf.nn.relu)
    l32 = tf.layers.conv2d(
        l31, 64, (3, 3), padding='same', activation=tf.nn.relu)

    l32_shape = tf.shape(l32)
    l32_resized = tf.image.resize_bilinear(
        l32, size=[2*l32_shape[1], 2*l32_shape[2]], align_corners=True)
    l33 = tf.concat([l32_resized, l8], axis=3)
    l34 = tf.layers.conv2d(
        l33, 32, (3, 3), padding='same', activation=tf.nn.relu)
    l35 = tf.layers.conv2d(
        l34, 32, (3, 3), padding='same', activation=tf.nn.relu)
    l36 = tf.layers.conv2d(
        l35, 32, (3, 3), padding='same', activation=tf.nn.relu)

    l36_shape = tf.shape(l36)
    l36_resized = tf.image.resize_bilinear(
        l36, size=[2*l36_shape[1], 2*l36_shape[2]], align_corners=True)
    l37 = tf.concat([l36_resized, l5], axis=3)
    l38 = tf.layers.conv2d(
        l37, 32, (3, 3), padding='same', activation=tf.nn.relu)
    l39 = tf.layers.conv2d(
        l38, 32, (3, 3), padding='same', activation=tf.nn.relu)
    l40 = tf.layers.conv2d(
        l39, 32, (3, 3), padding='same', activation=tf.nn.relu)

    l40_shape = tf.shape(l40)
    l40_resized = tf.image.resize_bilinear(
        l40, size=[2*l40_shape[1], 2*l40_shape[2]], align_corners=True)
    l41 = tf.concat([l40_resized, l2], axis=3)
    l42 = tf.layers.conv2d(
        l41, 32, (3, 3), padding='same', activation=tf.nn.relu)
    l43 = tf.layers.conv2d(
        l42, 32, (3, 3), padding='same', activation=tf.nn.relu)
    l44 = tf.layers.conv2d(
        l43, 32, (3, 3), padding='same', activation=tf.nn.relu)

    return tf.layers.conv2d(l44, 1, (1, 1), padding='same')
