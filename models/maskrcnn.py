import warnings
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock
from collections import OrderedDict
import torch
from efficientnet_pytorch import EfficientNet
from torchvision.models.detection.mask_rcnn import MaskRCNN

from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops

class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers):
      
        orig_return_layers = return_layers.copy()
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, md in model.named_children():
            if name != "_blocks":
                layers[name] = md
                if name in return_layers:
                    del return_layers[name]
                if not return_layers:
                    break
            else:
                for name_c, i in md.named_children():
                    
                    layers[name_c] = i
                    if name_c in return_layers:
                        del return_layers[name_c]
                    if not return_layers:
                        break

        super().__init__(layers)
        self.return_layers = orig_return_layers


    def forward(self, x):
        out = OrderedDict()

        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out

class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks,
    ):
        
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def b4_maskRCNN(num_classes = 2 + 80, pretrain = False):

	if pretrain:
	    back = EfficientNet.from_name('efficientnet-b4', include_top=False)
	    back_pre = EfficientNet.from_pretrained('efficientnet-b4')
	    back._conv_stem.weight = back_pre._conv_stem.weight
	    back._bn0.weight = back_pre._bn0.weight
	    for i, j in zip(back._blocks.children(), back_pre._blocks.children()):
	        for ii, jj in zip(i.children(), j.children()):
	            if hasattr(ii, "weight"):
	                ii.weight = jj.weight
	else:
	    back = EfficientNet.from_name('efficientnet-b4', include_top=False) 

	backbone = BackboneWithFPN(backbone = back, return_layers = {"28":'0', "29":'1', "30":'2', "31":'3'}, 
	                             in_channels_list=[272, 272, 448, 448], out_channels=256, extra_blocks=LastLevelMaxPool())

	return MaskRCNN(backbone, num_classes)


def b7_maskRCNN(num_classes = 2 + 80, pretrain = False):

	if pretrain:
	    back = EfficientNet.from_name('efficientnet-b7', include_top=False)
	    back_pre = EfficientNet.from_pretrained('efficientnet-b7')
	    back._conv_stem.weight = back_pre._conv_stem.weight
	    back._bn0.weight = back_pre._bn0.weight
	    for i, j in zip(back._blocks.children(), back_pre._blocks.children()):
	        for ii, jj in zip(i.children(), j.children()):
	            if hasattr(ii, "weight"):
	                ii.weight = jj.weight
	else:
	    back = EfficientNet.from_name('efficientnet-b7', include_top=False) 

	#b7
	backbone = BackboneWithFPN(backbone = back, return_layers = {"27":'0', "37":'1', "50":'2', "54":'3'}, 
	                             in_channels_list=[160, 224, 384, 640], out_channels=256, extra_blocks=LastLevelMaxPool())

	return MaskRCNN(backbone, num_classes)


from torchvision.models.resnet import resnext50_32x4d
def resNext50_maskRCNN(num_classes = 2 + 80, pretrain = False):
	
	backbone = resnext50_32x4d(pretrained=pretrain, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
	backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

	return MaskRCNN(backbone, num_classes)

from torchvision.models.resnet import resnet50
def resnet50_maskRCNN(num_classes = 2 + 80, pretrain = False):
	
	backbone = resnet50(pretrained=pretrain, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
	backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())

	return MaskRCNN(backbone, num_classes)

from torchvision.models.resnet import resnet34
def resnet34_maskRCNN(num_classes = 2 + 80, pretrain = False):
	
	backbone = resnet34(pretrained=pretrain, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
	backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, in_channels_list=[64, 128, 256, 512], out_channels=256, extra_blocks=LastLevelMaxPool())

	return MaskRCNN(backbone, num_classes)

from torchvision.models.resnet import resnet101
def resnet101_maskRCNN(num_classes = 2 + 80, pretrain = False):
	
	backbone = resnet101(pretrained=pretrain, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
	backbone = BackboneWithFPN(backbone = backbone, return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, in_channels_list=[256, 512, 1024, 2048], out_channels=256, extra_blocks=LastLevelMaxPool())
	
	return MaskRCNN(backbone, num_classes)


from torchvision.models.vgg import vgg19_bn
def vgg19_maskRCNN(num_classes = 2 + 80, pretrain = False):
	
	backbone = vgg19_bn(pretrained=pretrain, progress=True)
	backbone_fpn = BackboneWithFPN(backbone = backbone.features, return_layers = {"7":'0', "14":'1', "27":'2', "40":'3'}, in_channels_list=[128, 256, 512, 512], out_channels=256, extra_blocks=LastLevelMaxPool())
	
	return MaskRCNN(backbone_fpn, num_classes)


from torchvision.models.mobilenetv3 import mobilenet_v3_large
def mobile_maskRCNN(num_classes = 2 + 80, pretrain = False):
	
	backbone = mobilenet_v3_large(pretrained=pretrain, progress=True)
	backbone = BackboneWithFPN(backbone = backbone.features, return_layers = {"13":'0', "14":'1', "15":'2', "16":'3'}, in_channels_list=[160, 160, 160, 960], out_channels=256, extra_blocks=LastLevelMaxPool())
	
	return MaskRCNN(backbone, num_classes)