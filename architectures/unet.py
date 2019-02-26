import tensorflow as tf


def unet(x, is_training):
    def conv_block(net, filters, num_layer):
        for i in range(num_layer):
            net = tf.layers.conv2d(net, filters, (3, 3), (1, 1), 'same')
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

        skip = net

        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), 'same')

        return net, skip

    def deconv_block(net, skip, filters, num_layer):
        net = tf.layers.conv2d_transpose(
            net, int(filters / 2), (2, 2), (2, 2), 'same')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)

        net = tf.concat((net, skip), 3)

        for i in range(num_layer):
            net = tf.layers.conv2d(net, filters, (3, 3), (1, 1), 'same')
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

        return net

    net = x

    net, skip_1 = conv_block(net, 64, 1)
    net, skip_2 = conv_block(net, 128, 1)
    net, skip_3 = conv_block(net, 256, 2)
    net, skip_4 = conv_block(net, 512, 2)
    net, skip_5 = conv_block(net, 512, 2)
    net, skip_6 = conv_block(net, 512, 1)

    net = deconv_block(skip_6, skip_5, 512, 1)
    net = deconv_block(net, skip_4, 512, 1)
    net = deconv_block(net, skip_3, 256, 1)
    net = deconv_block(net, skip_2, 128, 1)
    net = deconv_block(net, skip_1, 64, 0)

    net = tf.layers.conv2d(net, 33, (3, 3), (1, 1), 'same')
    net = tf.nn.softmax(net)

    return net


def unet2(x, is_training):
    def conv_block(net, filters, num_layer):
        for i in range(num_layer):
            net = tf.layers.conv2d(net, filters, (3, 3), (1, 1), 'same')
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

        skip = net

        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), 'same')

        return net, skip

    def deconv_block(net, skip, filters, num_layer):
        net = tf.layers.conv2d_transpose(
            net, int(filters / 2), (2, 2), (2, 2), 'same')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)

        net = tf.concat((net, skip), 3)

        for i in range(num_layer):
            net = tf.layers.conv2d(net, filters, (3, 3), (1, 1), 'same')
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

        return net

    net = x

    net, skip_1 = conv_block(net, 64, 1)
    net, skip_2 = conv_block(net, 128, 1)
    net, skip_3 = conv_block(net, 256, 2)
    net, skip_4 = conv_block(net, 512, 2)
    net, skip_5 = conv_block(net, 512, 2)
    net, skip_6 = conv_block(net, 512, 1)

    net = tf.layers.dropout(skip_6, 0.5, training=is_training)

    net = deconv_block(net, skip_5, 512, 1)
    net = deconv_block(net, skip_4, 512, 1)
    net = deconv_block(net, skip_3, 256, 1)
    net = deconv_block(net, skip_2, 128, 1)
    net = deconv_block(net, skip_1, 64, 0)

    net = tf.layers.conv2d(net, 33, (1, 1), (1, 1), 'same')
    net = tf.nn.softmax(net)

    return net
