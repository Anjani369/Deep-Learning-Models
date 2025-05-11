import tensorflow as tf
from resnet_branch import build_resnet_branch
from sae_branch import build_sae_branch
from feature_fusion import fuse_features
from classifier_head import build_classifier

def build_brainnet_model(input_shape=(160, 210, 3)):
    input_img = tf.keras.Input(shape=input_shape)

    resnet_model = build_resnet_branch(input_shape)
    sae_model = build_sae_branch(input_shape)

    resnet_feat = resnet_model(input_img)
    sae_feat = sae_model(input_img)

    fused = fuse_features(resnet_feat, sae_feat)
    output = build_classifier(fused)

    return tf.keras.models.Model(inputs=input_img, outputs=output, name="BrainNet")
