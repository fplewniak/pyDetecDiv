"""
Deep-learning model combining ResNet18 and bidirectional LSTM
"""
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    # x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    # initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    # x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)

def create_model(n_classes: int) -> keras.Model:
    """
    Creates the model with the desired number of classes

    :param n_classes: the number of classes
    :return: the model
    """
    # resnet = resnet18(input_shape=(60, 60, 3), weights='imagenet', include_top=False)

    input_layer = keras.Input(shape=(None, 60, 60, 3))
    fold_out, fold_miniBatchSize = SequenceFoldingLayer((60, 60, 3))(input_layer)

    central_block = resnet18(fold_out)
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
