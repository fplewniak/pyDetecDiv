from tensorflow import keras

def create_model(n_classes):
    model = keras.Sequential()

    model.add(keras.layers.Input((60, 60, 3)))
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.2))
    # Maybe place a keras.layers.Flatten() layer here and find something for plugout in training. Might be possible
    # to specify that no special layer should be added before the output layer in the final model... for instance
    # using None instead of the name of the layer function...
    # Flattening here reduces the number of parameters a lot and allows a much faster training
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(n_classes, name="new_fc_"))
    model.add(keras.layers.Softmax())
    return model
