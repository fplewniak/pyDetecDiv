#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    03-Nov-2023 11:54:04

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input_unnormalized = keras.Input(shape=(None,224,224,3))
    input = SubtractConstantLayer((224,224,3), name="input_")(input_unnormalized)
    fold_out,fold_miniBatchSize = SequenceFoldingLayer((224,224,3))(input)
    conv1_7x7_s2_prepadded = layers.ZeroPadding2D(padding=((3,3),(3,3)))(fold_out)
    conv1_7x7_s2 = layers.Conv2D(64, (7,7), strides=(2,2), name="conv1_7x7_s2_")(conv1_7x7_s2_prepadded)
    conv1_relu_7x7 = layers.ReLU()(conv1_7x7_s2)
    pool1_3x3_s2_prepadded = layers.ZeroPadding2D(padding=((0,1),(0,1)))(conv1_relu_7x7)
    pool1_3x3_s2 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(pool1_3x3_s2_prepadded)
    CCNormLayer = layers.Lambda(lambda X: tf.nn.local_response_normalization(X, depth_radius=2.000000, bias=1.000000, alpha=0.000020, beta=0.750000))
    pool1_norm1 = CCNormLayer(pool1_3x3_s2)
    conv2_3x3_reduce = layers.Conv2D(64, (1,1), name="conv2_3x3_reduce_")(pool1_norm1)
    conv2_relu_3x3_reduce = layers.ReLU()(conv2_3x3_reduce)
    conv2_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(conv2_relu_3x3_reduce)
    conv2_3x3 = layers.Conv2D(192, (3,3), name="conv2_3x3_")(conv2_3x3_prepadded)
    conv2_relu_3x3 = layers.ReLU()(conv2_3x3)
    CCNormLayer = layers.Lambda(lambda X: tf.nn.local_response_normalization(X, depth_radius=2.000000, bias=1.000000, alpha=0.000020, beta=0.750000))
    conv2_norm2 = CCNormLayer(conv2_relu_3x3)
    pool2_3x3_s2_prepadded = layers.ZeroPadding2D(padding=((0,1),(0,1)))(conv2_norm2)
    pool2_3x3_s2 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(pool2_3x3_s2_prepadded)
    inception_3a_1x1 = layers.Conv2D(64, (1,1), name="inception_3a_1x1_")(pool2_3x3_s2)
    inception_3a_relu_1x1 = layers.ReLU()(inception_3a_1x1)
    inception_3a_3x3_reduce = layers.Conv2D(96, (1,1), name="inception_3a_3x3_reduce_")(pool2_3x3_s2)
    inception_3a_relu_3x3_reduce = layers.ReLU()(inception_3a_3x3_reduce)
    inception_3a_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_3a_relu_3x3_reduce)
    inception_3a_3x3 = layers.Conv2D(128, (3,3), name="inception_3a_3x3_")(inception_3a_3x3_prepadded)
    inception_3a_relu_3x3 = layers.ReLU()(inception_3a_3x3)
    inception_3a_5x5_reduce = layers.Conv2D(16, (1,1), name="inception_3a_5x5_reduce_")(pool2_3x3_s2)
    inception_3a_relu_5x5_reduce = layers.ReLU()(inception_3a_5x5_reduce)
    inception_3a_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_3a_relu_5x5_reduce)
    inception_3a_5x5 = layers.Conv2D(32, (5,5), name="inception_3a_5x5_")(inception_3a_5x5_prepadded)
    inception_3a_relu_5x5 = layers.ReLU()(inception_3a_5x5)
    inception_3a_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(pool2_3x3_s2)
    inception_3a_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_3a_pool_prepadded)
    inception_3a_pool_proj = layers.Conv2D(32, (1,1), name="inception_3a_pool_proj_")(inception_3a_pool)
    inception_3a_relu_pool_proj = layers.ReLU()(inception_3a_pool_proj)
    inception_3a_output = layers.Concatenate(axis=-1)([inception_3a_relu_1x1, inception_3a_relu_3x3, inception_3a_relu_5x5, inception_3a_relu_pool_proj])
    inception_3b_1x1 = layers.Conv2D(128, (1,1), name="inception_3b_1x1_")(inception_3a_output)
    inception_3b_relu_1x1 = layers.ReLU()(inception_3b_1x1)
    inception_3b_3x3_reduce = layers.Conv2D(128, (1,1), name="inception_3b_3x3_reduce_")(inception_3a_output)
    inception_3b_relu_3x3_reduce = layers.ReLU()(inception_3b_3x3_reduce)
    inception_3b_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_3b_relu_3x3_reduce)
    inception_3b_3x3 = layers.Conv2D(192, (3,3), name="inception_3b_3x3_")(inception_3b_3x3_prepadded)
    inception_3b_relu_3x3 = layers.ReLU()(inception_3b_3x3)
    inception_3b_5x5_reduce = layers.Conv2D(32, (1,1), name="inception_3b_5x5_reduce_")(inception_3a_output)
    inception_3b_relu_5x5_reduce = layers.ReLU()(inception_3b_5x5_reduce)
    inception_3b_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_3b_relu_5x5_reduce)
    inception_3b_5x5 = layers.Conv2D(96, (5,5), name="inception_3b_5x5_")(inception_3b_5x5_prepadded)
    inception_3b_relu_5x5 = layers.ReLU()(inception_3b_5x5)
    inception_3b_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_3a_output)
    inception_3b_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_3b_pool_prepadded)
    inception_3b_pool_proj = layers.Conv2D(64, (1,1), name="inception_3b_pool_proj_")(inception_3b_pool)
    inception_3b_relu_pool_proj = layers.ReLU()(inception_3b_pool_proj)
    inception_3b_output = layers.Concatenate(axis=-1)([inception_3b_relu_1x1, inception_3b_relu_3x3, inception_3b_relu_5x5, inception_3b_relu_pool_proj])
    pool3_3x3_s2_prepadded = layers.ZeroPadding2D(padding=((0,1),(0,1)))(inception_3b_output)
    pool3_3x3_s2 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(pool3_3x3_s2_prepadded)
    inception_4a_1x1 = layers.Conv2D(192, (1,1), name="inception_4a_1x1_")(pool3_3x3_s2)
    inception_4a_relu_1x1 = layers.ReLU()(inception_4a_1x1)
    inception_4a_3x3_reduce = layers.Conv2D(96, (1,1), name="inception_4a_3x3_reduce_")(pool3_3x3_s2)
    inception_4a_relu_3x3_reduce = layers.ReLU()(inception_4a_3x3_reduce)
    inception_4a_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4a_relu_3x3_reduce)
    inception_4a_3x3 = layers.Conv2D(208, (3,3), name="inception_4a_3x3_")(inception_4a_3x3_prepadded)
    inception_4a_relu_3x3 = layers.ReLU()(inception_4a_3x3)
    inception_4a_5x5_reduce = layers.Conv2D(16, (1,1), name="inception_4a_5x5_reduce_")(pool3_3x3_s2)
    inception_4a_relu_5x5_reduce = layers.ReLU()(inception_4a_5x5_reduce)
    inception_4a_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_4a_relu_5x5_reduce)
    inception_4a_5x5 = layers.Conv2D(48, (5,5), name="inception_4a_5x5_")(inception_4a_5x5_prepadded)
    inception_4a_relu_5x5 = layers.ReLU()(inception_4a_5x5)
    inception_4a_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(pool3_3x3_s2)
    inception_4a_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_4a_pool_prepadded)
    inception_4a_pool_proj = layers.Conv2D(64, (1,1), name="inception_4a_pool_proj_")(inception_4a_pool)
    inception_4a_relu_pool_proj = layers.ReLU()(inception_4a_pool_proj)
    inception_4a_output = layers.Concatenate(axis=-1)([inception_4a_relu_1x1, inception_4a_relu_3x3, inception_4a_relu_5x5, inception_4a_relu_pool_proj])
    inception_4b_1x1 = layers.Conv2D(160, (1,1), name="inception_4b_1x1_")(inception_4a_output)
    inception_4b_relu_1x1 = layers.ReLU()(inception_4b_1x1)
    inception_4b_3x3_reduce = layers.Conv2D(112, (1,1), name="inception_4b_3x3_reduce_")(inception_4a_output)
    inception_4b_relu_3x3_reduce = layers.ReLU()(inception_4b_3x3_reduce)
    inception_4b_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4b_relu_3x3_reduce)
    inception_4b_3x3 = layers.Conv2D(224, (3,3), name="inception_4b_3x3_")(inception_4b_3x3_prepadded)
    inception_4b_relu_3x3 = layers.ReLU()(inception_4b_3x3)
    inception_4b_5x5_reduce = layers.Conv2D(24, (1,1), name="inception_4b_5x5_reduce_")(inception_4a_output)
    inception_4b_relu_5x5_reduce = layers.ReLU()(inception_4b_5x5_reduce)
    inception_4b_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_4b_relu_5x5_reduce)
    inception_4b_5x5 = layers.Conv2D(64, (5,5), name="inception_4b_5x5_")(inception_4b_5x5_prepadded)
    inception_4b_relu_5x5 = layers.ReLU()(inception_4b_5x5)
    inception_4b_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4a_output)
    inception_4b_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_4b_pool_prepadded)
    inception_4b_pool_proj = layers.Conv2D(64, (1,1), name="inception_4b_pool_proj_")(inception_4b_pool)
    inception_4b_relu_pool_proj = layers.ReLU()(inception_4b_pool_proj)
    inception_4b_output = layers.Concatenate(axis=-1)([inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj])
    inception_4c_1x1 = layers.Conv2D(128, (1,1), name="inception_4c_1x1_")(inception_4b_output)
    inception_4c_relu_1x1 = layers.ReLU()(inception_4c_1x1)
    inception_4c_3x3_reduce = layers.Conv2D(128, (1,1), name="inception_4c_3x3_reduce_")(inception_4b_output)
    inception_4c_relu_3x3_reduce = layers.ReLU()(inception_4c_3x3_reduce)
    inception_4c_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4c_relu_3x3_reduce)
    inception_4c_3x3 = layers.Conv2D(256, (3,3), name="inception_4c_3x3_")(inception_4c_3x3_prepadded)
    inception_4c_relu_3x3 = layers.ReLU()(inception_4c_3x3)
    inception_4c_5x5_reduce = layers.Conv2D(24, (1,1), name="inception_4c_5x5_reduce_")(inception_4b_output)
    inception_4c_relu_5x5_reduce = layers.ReLU()(inception_4c_5x5_reduce)
    inception_4c_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_4c_relu_5x5_reduce)
    inception_4c_5x5 = layers.Conv2D(64, (5,5), name="inception_4c_5x5_")(inception_4c_5x5_prepadded)
    inception_4c_relu_5x5 = layers.ReLU()(inception_4c_5x5)
    inception_4c_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4b_output)
    inception_4c_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_4c_pool_prepadded)
    inception_4c_pool_proj = layers.Conv2D(64, (1,1), name="inception_4c_pool_proj_")(inception_4c_pool)
    inception_4c_relu_pool_proj = layers.ReLU()(inception_4c_pool_proj)
    inception_4c_output = layers.Concatenate(axis=-1)([inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj])
    inception_4d_1x1 = layers.Conv2D(112, (1,1), name="inception_4d_1x1_")(inception_4c_output)
    inception_4d_relu_1x1 = layers.ReLU()(inception_4d_1x1)
    inception_4d_3x3_reduce = layers.Conv2D(144, (1,1), name="inception_4d_3x3_reduce_")(inception_4c_output)
    inception_4d_relu_3x3_reduce = layers.ReLU()(inception_4d_3x3_reduce)
    inception_4d_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4d_relu_3x3_reduce)
    inception_4d_3x3 = layers.Conv2D(288, (3,3), name="inception_4d_3x3_")(inception_4d_3x3_prepadded)
    inception_4d_relu_3x3 = layers.ReLU()(inception_4d_3x3)
    inception_4d_5x5_reduce = layers.Conv2D(32, (1,1), name="inception_4d_5x5_reduce_")(inception_4c_output)
    inception_4d_relu_5x5_reduce = layers.ReLU()(inception_4d_5x5_reduce)
    inception_4d_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_4d_relu_5x5_reduce)
    inception_4d_5x5 = layers.Conv2D(64, (5,5), name="inception_4d_5x5_")(inception_4d_5x5_prepadded)
    inception_4d_relu_5x5 = layers.ReLU()(inception_4d_5x5)
    inception_4d_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4c_output)
    inception_4d_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_4d_pool_prepadded)
    inception_4d_pool_proj = layers.Conv2D(64, (1,1), name="inception_4d_pool_proj_")(inception_4d_pool)
    inception_4d_relu_pool_proj = layers.ReLU()(inception_4d_pool_proj)
    inception_4d_output = layers.Concatenate(axis=-1)([inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj])
    inception_4e_1x1 = layers.Conv2D(256, (1,1), name="inception_4e_1x1_")(inception_4d_output)
    inception_4e_relu_1x1 = layers.ReLU()(inception_4e_1x1)
    inception_4e_3x3_reduce = layers.Conv2D(160, (1,1), name="inception_4e_3x3_reduce_")(inception_4d_output)
    inception_4e_relu_3x3_reduce = layers.ReLU()(inception_4e_3x3_reduce)
    inception_4e_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4e_relu_3x3_reduce)
    inception_4e_3x3 = layers.Conv2D(320, (3,3), name="inception_4e_3x3_")(inception_4e_3x3_prepadded)
    inception_4e_relu_3x3 = layers.ReLU()(inception_4e_3x3)
    inception_4e_5x5_reduce = layers.Conv2D(32, (1,1), name="inception_4e_5x5_reduce_")(inception_4d_output)
    inception_4e_relu_5x5_reduce = layers.ReLU()(inception_4e_5x5_reduce)
    inception_4e_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_4e_relu_5x5_reduce)
    inception_4e_5x5 = layers.Conv2D(128, (5,5), name="inception_4e_5x5_")(inception_4e_5x5_prepadded)
    inception_4e_relu_5x5 = layers.ReLU()(inception_4e_5x5)
    inception_4e_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_4d_output)
    inception_4e_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_4e_pool_prepadded)
    inception_4e_pool_proj = layers.Conv2D(128, (1,1), name="inception_4e_pool_proj_")(inception_4e_pool)
    inception_4e_relu_pool_proj = layers.ReLU()(inception_4e_pool_proj)
    inception_4e_output = layers.Concatenate(axis=-1)([inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj])
    pool4_3x3_s2_prepadded = layers.ZeroPadding2D(padding=((0,1),(0,1)))(inception_4e_output)
    pool4_3x3_s2 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(pool4_3x3_s2_prepadded)
    inception_5a_1x1 = layers.Conv2D(256, (1,1), name="inception_5a_1x1_")(pool4_3x3_s2)
    inception_5a_relu_1x1 = layers.ReLU()(inception_5a_1x1)
    inception_5a_3x3_reduce = layers.Conv2D(160, (1,1), name="inception_5a_3x3_reduce_")(pool4_3x3_s2)
    inception_5a_relu_3x3_reduce = layers.ReLU()(inception_5a_3x3_reduce)
    inception_5a_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_5a_relu_3x3_reduce)
    inception_5a_3x3 = layers.Conv2D(320, (3,3), name="inception_5a_3x3_")(inception_5a_3x3_prepadded)
    inception_5a_relu_3x3 = layers.ReLU()(inception_5a_3x3)
    inception_5a_5x5_reduce = layers.Conv2D(32, (1,1), name="inception_5a_5x5_reduce_")(pool4_3x3_s2)
    inception_5a_relu_5x5_reduce = layers.ReLU()(inception_5a_5x5_reduce)
    inception_5a_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_5a_relu_5x5_reduce)
    inception_5a_5x5 = layers.Conv2D(128, (5,5), name="inception_5a_5x5_")(inception_5a_5x5_prepadded)
    inception_5a_relu_5x5 = layers.ReLU()(inception_5a_5x5)
    inception_5a_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(pool4_3x3_s2)
    inception_5a_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_5a_pool_prepadded)
    inception_5a_pool_proj = layers.Conv2D(128, (1,1), name="inception_5a_pool_proj_")(inception_5a_pool)
    inception_5a_relu_pool_proj = layers.ReLU()(inception_5a_pool_proj)
    inception_5a_output = layers.Concatenate(axis=-1)([inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj])
    inception_5b_1x1 = layers.Conv2D(384, (1,1), name="inception_5b_1x1_")(inception_5a_output)
    inception_5b_relu_1x1 = layers.ReLU()(inception_5b_1x1)
    inception_5b_3x3_reduce = layers.Conv2D(192, (1,1), name="inception_5b_3x3_reduce_")(inception_5a_output)
    inception_5b_relu_3x3_reduce = layers.ReLU()(inception_5b_3x3_reduce)
    inception_5b_3x3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_5b_relu_3x3_reduce)
    inception_5b_3x3 = layers.Conv2D(384, (3,3), name="inception_5b_3x3_")(inception_5b_3x3_prepadded)
    inception_5b_relu_3x3 = layers.ReLU()(inception_5b_3x3)
    inception_5b_5x5_reduce = layers.Conv2D(48, (1,1), name="inception_5b_5x5_reduce_")(inception_5a_output)
    inception_5b_relu_5x5_reduce = layers.ReLU()(inception_5b_5x5_reduce)
    inception_5b_5x5_prepadded = layers.ZeroPadding2D(padding=((2,2),(2,2)))(inception_5b_relu_5x5_reduce)
    inception_5b_5x5 = layers.Conv2D(128, (5,5), name="inception_5b_5x5_")(inception_5b_5x5_prepadded)
    inception_5b_relu_5x5 = layers.ReLU()(inception_5b_5x5)
    inception_5b_pool_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(inception_5a_output)
    inception_5b_pool = layers.MaxPool2D(pool_size=(3,3), strides=(1,1))(inception_5b_pool_prepadded)
    inception_5b_pool_proj = layers.Conv2D(128, (1,1), name="inception_5b_pool_proj_")(inception_5b_pool)
    inception_5b_relu_pool_proj = layers.ReLU()(inception_5b_pool_proj)
    inception_5b_output = layers.Concatenate(axis=-1)([inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj])
    pool5_7x7_s1 = layers.GlobalAveragePooling2D(keepdims=True)(inception_5b_output)
    unfold = SequenceUnfoldingLayer((1,1,1024))(pool5_7x7_s1, fold_miniBatchSize)
    unfoldperm = layers.TimeDistributed(layers.Permute((3,2,1)))(unfold)
    flatten = layers.TimeDistributed(layers.Flatten())(unfoldperm)
    flatten_bilstm_input = flatten
    bilstm = layers.Bidirectional(layers.LSTM(150, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=False), name="bilstm_")(flatten_bilstm_input)
    drop = layers.Dropout(0.500000)(bilstm)
    fc = layers.Dense(6, name="fc_")(drop)
    softmax = layers.Softmax()(fc)
    classification = softmax

    model = keras.Model(inputs=[input_unnormalized], outputs=[input, classification])
    return model

## Helper layers:

class SequenceFoldingLayer(tf.keras.layers.Layer):
    def __init__(self, dataShape, name=None):
        super(SequenceFoldingLayer, self).__init__(name=name)
        self.dataShape = dataShape;

    def call(self, input):
        # Two outputs: Y and batchSize
        shape = tf.shape(input)
        return tf.reshape(input, (-1,) + self.dataShape), shape[0]


class SequenceUnfoldingLayer(tf.keras.layers.Layer):
    def __init__(self, dataShape, name=None):
        super(SequenceUnfoldingLayer, self).__init__(name=name)
        self.dataShape = dataShape;

    def call(self, X, batchSize):
        return tf.reshape(X, (batchSize, -1) + self.dataShape)


class SubtractConstantLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(SubtractConstantLayer, self).__init__(name=name)
        self.const = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        return input - self.const
