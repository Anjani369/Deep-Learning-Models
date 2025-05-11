import tensorflow as tf
from tensorflow.keras import layers

def fuse_features(resnet_feat, sae_feat):
    sae_feat_padded = layers.Concatenate()([sae_feat, layers.ZeroPadding1D((0, 256))(sae_feat)])
    return layers.Average()([resnet_feat, sae_feat_padded])
