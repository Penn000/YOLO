import tensorflow as tf
import numpy as np
import cv2

class YOLO_v1:

    def conv_layer(self, x, filters, size, stride, name):
        n_channels = int(x.get_shape()[3])
        with tf.variable_scope(name):
            weight = tf.get_variable(
                'weight', 
                shape=(size, size, n_channels, filters), 
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            biases = tf.get_shape(
                'biases',
                shape=(filters),
                initializer=tf.constant_initializer(0.1)
            )
            layer = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='VALID', name=name) + biases
            relu = tf.nn.leaky_relu(layer, alpha=self.alpha, name=name+'_relu')
        return relu
    
    def pooling_layer(self, x, size, stride, name):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def fc_layer(self, x, in_size, out_size, name, linear=False):
        with tf.variable_scope(name):
            weight = tf.get_variable(
                'weight', 
                shape=(in_size, out_size), 
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            biases = tf.get_shape(
                'biases',
                shape=(out_size),
                initializer=tf.constant_initializer(0.1)
            )
            layer = tf.nn.xw_plus_b(x, weight, biases, name=name)
            if linear: return layer
            relu = tf.nn.leaky_relu(layer, alpha=self.alpha, name=name+'_relu')
        return relu

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, 448, 448, 3])

        # input: 448x448x3
        # output: 112x112x192
        self.conv1_1 = self.conv_layer(x, 64, 7, 2, 'conv1_1')
        self.pool1 = self.pooling_layer(self.conv1_1, 2, 2, 'pool1')

        # input: 112x112x192
        # output: 56x56x256
        self.conv2_1 = self.conv_layer(self.pool1, 192, 3, 1, 'conv2_1')
        self.pool2 = self.pooling_layer(self.conv2_1, 2, 2, 'pool2')

        # input: 56x56x256
        # output: 28x28x512
        self.conv3_1 = self.conv_layer(self.pool2, 128, 1, 1, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 3, 1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 1, 1, 'conv3_3')
        self.conv3_4 = self.conv_layer(self.conv3_3, 512, 3, 1, 'conv3_4')
        self.pool3 = self.pooling_layer(self.conv3_4, 2, 2, 'pool9')

        # input: 28x28x512
        # output: 14x14x1024
        self.conv4_1 = self.conv_layer(self.pool3, 256, 1, 1, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 3, 1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 256, 1, 1, 'conv4_3')
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 3, 1, 'conv4_4')
        self.conv4_5 = self.conv_layer(self.conv4_4, 256, 1, 1, 'conv4_5')
        self.conv4_6 = self.conv_layer(self.conv4_5, 512, 3, 1, 'conv4_6')
        self.conv4_7 = self.conv_layer(self.conv4_6, 256, 1, 1, 'conv4_7')
        self.conv4_8 = self.conv_layer(self.conv4_7, 512, 3, 1, 'conv4_8')
        self.conv4_9 = self.conv_layer(self.conv4_8, 512, 1, 1, 'conv4_9')
        self.conv4_10 = self.conv_layer(self.conv4_9, 1024, 3, 1,'conv4_10')
        self.pool4 = self.pooling_layer(self.conv4_10, 2, 2, 'pool4')

        # input: 14x14x1024
        # output: 7x7x1024
        self.conv5_1 = self.conv_layer(self.pool4, 512, 1, 1, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 1024, 3, 1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 1, 1, 'conv5_3')
        self.conv5_4 = self.conv_layer(self.conv5_3, 1024, 3, 1, 'conv5_4')
        self.conv5_5 = self.conv_layer(self.conv5_4, 1024, 3, 1, 'conv5_5')
        self.conv5_6 = self.conv_layer(self.conv5_5, 1024, 3, 2, 'conv5_6')

        # input: 7x7x1024
        # output: 7x7x1024
        self.conv6_1 = self.conv_layer(self.conv5_1, 1024, 3, 1, 'conv6_1')
        self.conv6_2 = self.conv_layer(self.conv6_1, 1024, 3, 1, 'conv6_2')

        # flatten
        # input: 7x7x1024
        # output: 50176
        self.flat = tf.layers.flatten(
            tf.transpose(self.conv6_2, (0, 3, 1, 2)),
            name='flatten'
        )

        # input: 50176
        # output: 512
        self.fc7 = self.fc_layer(self.flat, 50176, 512, 'fc7')

        # input: 512
        # output: 4096
        self.fc8 = self.fc_layer(self.fc7, 512, 4096, 'fc8')

        # input: 4096
        # output: 1470
        self.fc7 = self.fc_layer(self.fc8, 4096, 1470, 'fc9', linear=True)

    def nms(self, x):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(x[0: 980], (7, 7, 20))
        scales = np.reshape(x[980: 1078], (7, 7, 2))
        bboxes = np.reshape(x[1078:], (7, 7, 2, 4))