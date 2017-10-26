from lib.models.resnet_model import ResNet
import tensorflow.contrib.slim as slim
from lib.models.layers import *

class WideResNet_28_10(ResNet):

    def post_pool_operations(self, x):
        return x
