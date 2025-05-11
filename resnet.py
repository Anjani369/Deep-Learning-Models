import tensorflow as tf
from tensorflow.keras import layers, models

def build_resnet_branch(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg'
    )

    x = base_model.output
    for _ in range(3):  # Add 3 extra residual blocks
        shortcut = x
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

    resnet_output = layers.Dense(512, activation='relu')(x)
    return models.Model(inputs=base_model.input, outputs=resnet_output, name="ResNetBranch")
