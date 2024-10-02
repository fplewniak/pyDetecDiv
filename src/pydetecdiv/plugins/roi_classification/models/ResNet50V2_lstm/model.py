"""
Deep-learning model combining ResNet50V2 pre-trained on ImageNet and bidirectional LSTM
"""
import tensorflow as tf
from tensorflow import keras


def create_model(n_classes: int) -> keras.Model:
    """
    Creates the model with the desired number of classes

    :param n_classes: the number of classes
    :return: the model
    """
    resnet = keras.applications.ResNet50V2(input_shape=(60, 60, 3), weights='imagenet', include_top=False)

    input_layer = keras.Input(shape=(None, 60, 60, 3))
    fold_out, fold_miniBatchSize = SequenceFoldingLayer((60, 60, 3))(input_layer)

    central_block = resnet(fold_out, training=True)
    plugout = keras.layers.GlobalAveragePooling2D()(central_block)

    # unfolding layer should be the same size as the output from the previous block
    unfold = SequenceUnfoldingLayer((1, 1, plugout.shape[1]))(plugout, fold_miniBatchSize)
    unfoldperm = keras.layers.TimeDistributed(keras.layers.Permute((3, 2, 1)))(unfold)
    flatten = keras.layers.TimeDistributed(keras.layers.Flatten())(unfoldperm)
    flatten_bilstm_input = flatten
    bilstm = keras.layers.Bidirectional(keras.layers.LSTM(150, activation='tanh', recurrent_activation='sigmoid',
                                              return_sequences=True, return_state=False),
                                  name="bilstm_")(flatten_bilstm_input)
    drop = keras.layers.Dropout(0.500000)(bilstm)
    fc = keras.layers.Dense(n_classes, name="fc_")(drop)
    softmax = keras.layers.Softmax()(fc)
    classification = softmax

    model = keras.Model(inputs=[input_layer], outputs=[classification])
    return model


## Helper layers:

class SequenceFoldingLayer(keras.layers.Layer):
    """
    a layer to fold a sequence of images to feed the resnet
    """

    def __init__(self, data_shape: tuple[int], name: str = None) -> None:
        super().__init__(name=name)
        # print(dataShape)
        self.dataShape: tuple[int] = data_shape

    def call(self, input_layer: tf.Tensor) -> tuple[tf.Tensor, int]:
        """

        :param input_layer:
        :return:
        """
        # Two outputs: Y and batchSize
        shape = tf.shape(input_layer)
        return tf.reshape(input_layer, (-1,) + self.dataShape), shape[0]


class SequenceUnfoldingLayer(keras.layers.Layer):
    """
    a layer to unfold a sequence of images to feed the LSTM
    """

    def __init__(self, data_shape: tuple[int], name: str = None) -> None:
        super().__init__(name=name)
        self.dataShape = data_shape

    def call(self, input_tensor: tf.Tensor | list[tf.Tensor] | tuple[tf.Tensor], batch_size: int = 4) -> tf.Tensor:
        """

        :param input_tensor:
        :param batch_size:
        :return:
        """
        # print(X.shape)
        return tf.reshape(input_tensor, (batch_size, -1) + self.dataShape)
