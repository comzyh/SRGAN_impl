import tensorflow as tf


def SRResNet(images, training, reuse=False, residual_blocks_num=16):
    """
    Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., … Shi Twitter, W. (n.d.).
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.
    """
    with tf.variable_scope("SRResNet"):
        # k9n64s1
        x = tf.layers.conv2d(
            inputs=images, filters=64, kernel_size=(9, 9), strides=(1, 1),
            padding='same', use_bias=False,
        )
        x = tf.nn.leaky_relu(x)
        skip_all_residual_blocks = x
        # residual blocks

        for i in range(residual_blocks_num):
            skip = x
            x = tf.layers.conv2d(
                inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1),
                padding='same', use_bias=False,
                name='k3n64s1/{}/conv1'.format(i)
            )
            x = tf.layers.batch_normalization(
                inputs=x, axis=3, fused=True, scale=True,
                training=training,
                name='k3n64s1/{}/bn1'.format(i)
            )
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(
                inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1),
                padding='same', use_bias=False,
                name='k3n64s1/{}/conv2'.format(i)
            )
            x = tf.layers.batch_normalization(
                inputs=x, axis=3, fused=True, scale=True,
                training=training,
                name='k3n64s1/{}/bn2'.format(i)
            )
            x = skip + x
        # k3n64s1
        x = tf.layers.conv2d(
            inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False,
            name='k3n64s1/{}/conv1'.format(i + 1)
        )
        x = tf.layers.batch_normalization(
            inputs=x, axis=3, fused=True, scale=True,
            training=training,
            name='k3n64s1/{}/bn1'.format(i + 1)
        )
        x = skip_all_residual_blocks + x

        # k3n256s1
        for i in range(1):  # we use 2x upscaleing
            # 256 = 64 * 2 * 2
            x = tf.layers.conv2d(
                inputs=x, filters=256, kernel_size=(3, 3), strides=(1, 1),
                padding='same', use_bias=False,
                name='k3n256s1/{}/conv1'.format(i)
            )
            # reduce channel, increase size
            # https://www.tensorflow.org/api_docs/python/tf/depth_to_space
            x = tf.depth_to_space(x, 2)
            x = tf.nn.leaky_relu(x)

        # k9n3s1
        x = tf.layers.conv2d(
            inputs=x, filters=3, kernel_size=(9, 9), strides=(1, 1),
            padding='same', use_bias=False,
            name='k9n3s1'
        )
    return x
