#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    03-Nov-2023 11:54:04

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(n_classes=6):
    convnext = keras.applications.ConvNeXtTiny(input_shape=(60,60,3), weights='imagenet', include_top=False)

    input = keras.Input(shape=(None,60,60,3))
    fold_out, fold_miniBatchSize = SequenceFoldingLayer((60,60,3))(input)

    central_block = convnext(fold_out, training=True)
    plugout = keras.layers.GlobalAveragePooling2D()(central_block)

    # unfolding layer should be the same size as the output from the previous block
    unfold = SequenceUnfoldingLayer((1,1,plugout.shape[1]))(plugout, fold_miniBatchSize)
    unfoldperm = layers.TimeDistributed(layers.Permute((3,2,1)))(unfold)
    flatten = layers.TimeDistributed(layers.Flatten())(unfoldperm)
    flatten_bilstm_input = flatten
    bilstm = layers.Bidirectional(layers.LSTM(150, activation='tanh', recurrent_activation='sigmoid',
                                              return_sequences=True, return_state=False),
                                  name="bilstm_")(flatten_bilstm_input)
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
