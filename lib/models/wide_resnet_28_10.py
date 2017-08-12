from lib.models.resnet_model import ResNet


class WideResNet_28_10(ResNet):

    def add_fc_layers(self, x):
        return x
