"""
Main function to build sca_iqa_net.
"""
from tensorflow.keras.layers import Input, Dense, Average, GlobalAveragePooling2D, Dropout, Conv2D, Reshape
from tensorflow.keras.models import Model
from image_quality.models.prediction_model_contrast_sensitivity import channel_spatial_attention
from backbone.resnet50 import ResNet50

from backbone.vgg16 import VGG16
from backbone.densenet import DenseNet121
from backbone.efficientnet import EfficientNetB0, EfficientNetB4
import tensorflow as tf


def sca_iqa_net(n_quality_levels, input_shape=(None, None, 3), backbone='resnet50'):
    """
    Build sca_iqa_net
    :param n_quality_levels: 1 for MOS prediction and 5 for score distribution
    :param input_shape: image input shape, keep as unspecifized
    :param backbone: backbone networks (resnet50/18/152v2, resnest, vgg16, etc.)
    :return: sca_iqa_net model
    """
    inputs = Input(shape=input_shape)
    return_last_map = True
    if backbone == 'resnet50':
        backbone_model = ResNet50(inputs,
                                  return_last_map=return_last_map)
    elif backbone == 'densenet121':
        backbone_model = DenseNet121(inputs, return_last_map=return_last_map)
    elif backbone == 'vgg16':
        backbone_model = VGG16(inputs,
                                  return_last_map=return_last_map)
    elif backbone == 'efficientnetb0':
        backbone_model = EfficientNetB0(inputs)
    elif backbone == 'efficientnetb4':
        backbone_model = EfficientNetB4(inputs)
    else:
        backbone_model = None

    outputs = backbone_model.output

    outputs = channel_spatial_attention(outputs, n_quality_levels, 'P')

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


from image_quality.misc.model_flops import get_flops
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    input_shape = [None, None, 3]
    # input_shape = [768, 1024, 3]
    # input_shape = [500, 500, 3]
    model = sca_iqa_net(n_quality_levels=5, input_shape=input_shape, backbone='resnet50')
    model.summary()