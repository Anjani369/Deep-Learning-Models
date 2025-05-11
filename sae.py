import tensorflow as tf
from tensorflow.keras import layers, models

def build_sae_branch(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    sae_output = layers.Dense(256, activation='relu')(x)

    return models.Model(inputs=inputs, outputs=sae_output, name="SAEBranch")
