#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    03-Nov-2023 11:54:04

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(n_classes):
    input = keras.Input(shape=(None,60,60,3))
    fold_out, fold_miniBatchSize = SequenceFoldingLayer((60,60,3))(input)
    conv2d_1 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(fold_out)
    max_pool_1 = keras.layers.MaxPooling2D(2, 2)(conv2d_1)
    dropout_1 = keras.layers.Dropout(0.2)(max_pool_1)
    conv2d_2 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout_1)
    max_pool_2 = keras.layers.MaxPooling2D(2, 2)(conv2d_2)
    dropout_2 = keras.layers.Dropout(0.2)(max_pool_2)
    conv2d_3 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout_2)
    max_pool_3 = keras.layers.MaxPooling2D(2, 2)(conv2d_3)
    flatten_1 = keras.layers.Flatten()(max_pool_3)
    dense_1 = keras.layers.Dense(720, activation='relu')(flatten_1)
    dropout_3 = keras.layers.Dropout(0.2)(dense_1)
    # unfolding layer should be the same size as the output from the previous block
    unfold = SequenceUnfoldingLayer((1,1,720))(dropout_3, fold_miniBatchSize)
    unfoldperm = layers.TimeDistributed(layers.Permute((3,2,1)))(unfold)
    flatten = layers.TimeDistributed(layers.Flatten())(unfoldperm)
    flatten_bilstm_input = flatten
    bilstm = layers.Bidirectional(layers.LSTM(150, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=False), name="bilstm_")(flatten_bilstm_input)
    drop = layers.Dropout(0.500000)(bilstm)
    fc = layers.Dense(n_classes, name="fc_")(drop)
    softmax = layers.Softmax()(fc)
    classification = softmax

    model = keras.Model(inputs=[input], outputs=[classification])
    return model

## Helper layers:

class SequenceFoldingLayer(tf.keras.layers.Layer):
    def __init__(self, dataShape, name=None):
        super(SequenceFoldingLayer, self).__init__(name=name)
        # print(dataShape)
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
        # print(X.shape)
        return tf.reshape(X, (batchSize, -1) + self.dataShape)
