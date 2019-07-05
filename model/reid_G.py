import torch.nn as nn
import numpy as np
import torchvision
from torch.nn import init
from torch.nn import functional as F


affine_par = True

# __factory = {
#     # 18: [2,2,2,2],
#     # 34: [3,4,6,3],
#     50: [3, 4, 6, 3],
#     101: [3, 4, 23, 3],
#     152: [3, 8, 36, 3],
# }

def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        ## usage : self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out



class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    # def __init__(self, block, layers, num_classes):
    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0):
        ## used in : model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
        ## resnet101 : layers:[3,4,23,3]
        ## resnet101:
        ## conv1 : 7x7,64,stride + bn + relu + 3x3 maxpool;
        ## conv2 : [1x1 64, 3x3 64, 1x1 256] x 3
        ## conv3 : [1x1 128, 3x3 128, 1x1 512] x 4
        ## conv4 : [1x1 256, 3x3 256, 1x1 1024] x 23
        ## conv5 : [1x1 512, 3x3 512, 1x1 2048] x 3
        ## 1 x 1 average pool, 1000-d fc, softmax

        # self.inplanes = 64
        super(ResNet, self).__init__() ## basic operations
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.num_triplet_features = num_triplet_features
            self.l2norm = Normalize(2)

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes

            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes > 0:
                self.classifier_1 = nn.Linear(self.num_features, self.num_classes)
                self.classifier_2 = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier_1.weight, std=0.001)
                init.constant_(self.classifier_1.bias, 0)
                init.normal_(self.classifier_2.weight, std=0.001)
                init.constant_(self.classifier_2.bias, 0)

        if not self.pretrained:
            self.reset_params()


        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        # Fix layers [conv1 ~ layer2]
        # fixed_names = []
        # for name, module in self.base._modules.items():
        #     if name == "layer3":
        #         # assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
        #         break
        #     fixed_names.append(name)
        #     for param in module.parameters():
        #         param.requires_grad = False

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    # def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
    #     for i in downsample._modules['1'].parameters():
    #         i.requires_grad = False
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, dilation=dilation))
    #
    #     return nn.Sequential(*layers) # pass the sub instance as params of Sequential one by one

    # def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
    #     return block(inplanes, dilation_series, padding_series, num_classes)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x1 = self.layer5(x)
    #     x2 = self.layer6(x)
    #     return x1, x2


    def forward(self, x, output_feature=None):
        ## parameters :
        ## x : input tensor
        ## return :
        ## dict : a dict include features in different stage , resnet(conv1-->layer4) -> pool5 --> fc(tgt_feat) --> clf1,clf2
        ## ie. features_dict{'layer4' : tensor, 'pool5' : tensor, 'tgt_feat': tensor, clf1' : tensor, 'clf2' : tensor}
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)
        features_dict = {}
        # if self.cut_at_pooling:
        #     return x

        features_dict['layer4'] = x
        if self.cut_at_pooling:
            return features_dict

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        # if output_feature == 'pool5':
        #     x = F.normalize(x)
        #     return x

        features_dict['pool5'] = F.normalize(x)
        if output_feature == 'pool5' :
            # x = F.normalize(x)
            # features_dict['pool5'] = x
            return features_dict

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)
            features_dict['tgt_feat'] = tgt_feat
            if output_feature == 'tgt_feat':
                return features_dict

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x1 = self.classifier_1(x)
            x2 = self.classifier_2(x)
            features_dict['clf1'] = x1
            features_dict['clf2'] = x2
        return  features_dict

    def get_1x_lr_params_NOscale(self):
        # TODO(tb) implement this function to get the params of reid resnet
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """121
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        # b.append(self.layer5.parameters())
        # b.append(self.layer6.parameters())
        #b.append(self.layer7.parameters())
        b.append(self.classifier_1.parameters())
        b.append(self.classifier_2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

# def Res_Deeplab(num_classes=21):
#     model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
#     return model


