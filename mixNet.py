'''
MixNet From <MixConv: Mixed Depthwise Convolutional Kernels>, arXiv:1907.09595.
Ref: https://github.com/romulus0914/MixNet-PyTorch/blob/master/mixnet.py
'''

import torch
import torch.nn as nn
from collections import OrderedDict


def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def _split_channels(channels, num_groups):
    # split the channels into `num_groups` groups
    split_channels = [channels//num_groups] * num_groups
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_num, out_channels=in_channels_num // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(in_channels=in_channels_num // reduction_ratio, out_channels=in_channels_num, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_sizes)
        self.split_in_channels = _split_channels(in_channels, self.num_groups)
        self.split_out_channels = _split_channels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                in_channels=self.split_in_channels[i],
                out_channels=self.split_out_channels[i],
                kernel_size=kernel_sizes[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class MixConv(nn.Module):
    def __init__(self, channels, kernel_sizes, stride):
        super(MixConv, self).__init__()

        self.num_groups = len(kernel_sizes)
        self.split_channels = _split_channels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                in_channels=self.split_channels[i],
                out_channels=self.split_channels[i],
                kernel_size=kernel_sizes[i],
                stride=stride,
                padding=kernel_sizes[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, exp_kernel_sizes, project_kernel_sizes,
                stride, exp_ratio, use_swish, se_ratio, BN_momentum, BN_eps):

        super(Bottleneck, self).__init__()

        exp_size = in_channels * exp_ratio
        use_se = se_ratio >= 1
        self.use_residual = (stride == 1 and in_channels == out_channels)

        conv = []
        if exp_ratio > 1:
            # expansion phase
            conv.append(nn.Sequential(
                GroupedConv2d(in_channels=in_channels, out_channels=exp_size, kernel_sizes=exp_kernel_sizes),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum, eps=BN_eps),
                Swish() if use_swish else nn.ReLU()
            ))

        # depthwise convolution phase
        conv.append(nn.Sequential(
            MixConv(channels=exp_size, kernel_sizes=kernel_sizes, stride=stride),
            nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum, eps=BN_eps),
            Swish() if use_swish else nn.ReLU()
        ))

        if use_se:
            # squeeze and excite
            conv.append(SEModule(in_channels_num=exp_size, reduction_ratio=se_ratio))

        # projection phase
        conv.append(nn.Sequential(
            GroupedConv2d(in_channels=exp_size, out_channels=out_channels, kernel_sizes=project_kernel_sizes),
            nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels, momentum=BN_momentum, eps=BN_eps))]))
                            if self.use_residual else nn.BatchNorm2d(num_features=out_channels, momentum=BN_momentum, eps=BN_eps)
        ))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MixNet(nn.Module):

    def __init__(self, mode='S', num_classes=1000, input_size=224, width_multiplier=1.0, BN_momentum=0.1, BN_eps=1e-3, zero_gamma=False):
        super(MixNet, self).__init__()

        mode = mode.lower()
        assert mode in ['s', 'm', 'l']
        s = 2
        if input_size == 32 or input_size == 56:
            # using cifar-10, cifar-100, Tiny-ImageNet or Downsampled ImageNet
            s = 1

        if mode == 's':
            # out_channels, kernel_sizes, exp_kernel_sizes, project_kernel_sizes, stride, exp_ratio, use_swish, se_ratio
            configs = [
                (16,  [3],              [1],    [1],    1, 1, False, 0),
                (24,  [3],              [1, 1], [1, 1], s, 6, False, 0),
                (24,  [3],              [1, 1], [1, 1], 1, 3, False, 0),
                (40,  [3, 5, 7],        [1],    [1],    2, 6, True,  2),
                (40,  [3, 5],           [1, 1], [1, 1], 1, 6, True,  2),
                (40,  [3, 5],           [1, 1], [1, 1], 1, 6, True,  2),
                (40,  [3, 5],           [1, 1], [1, 1], 1, 6, True,  2),
                (80,  [3, 5, 7],        [1],    [1, 1], 2, 6, True,  4),
                (80,  [3, 5],           [1],    [1, 1], 1, 6, True,  4),
                (80,  [3, 5],           [1],    [1, 1], 1, 6, True,  4),
                (120, [3, 5, 7],        [1, 1], [1, 1], 1, 6, True,  2),
                (120, [3, 5, 7, 9],     [1, 1], [1, 1], 1, 3, True,  2),
                (120, [3, 5, 7, 9],     [1, 1], [1, 1], 1, 3, True,  2),
                (200, [3, 5, 7, 9, 11], [1],    [1],    2, 6, True,  2),
                (200, [3, 5, 7, 9],     [1],    [1, 1], 1, 6, True,  2),
                (200, [3, 5, 7, 9],     [1],    [1, 1], 1, 6, True,  2)
            ]
            first_channels_num = 16
            dropout = 0.2
        elif mode == 'm' or mode == 'l':
            configs = [
                (24,  [3],          [1],    [1],    1, 1, False, 0),
                (32,  [3, 5, 7],    [1, 1], [1, 1], s, 6, False, 0),
                (32,  [3],          [1, 1], [1, 1], 1, 3, False, 0),
                (40,  [3, 5, 7, 9], [1],    [1],    2, 6, True,  2),
                (40,  [3, 5],       [1, 1], [1, 1], 1, 6, True,  2),
                (40,  [3, 5],       [1, 1], [1, 1], 1, 6, True,  2),
                (40,  [3, 5],       [1, 1], [1, 1], 1, 6, True,  2),
                (80,  [3, 5, 7],    [1],    [1],    2, 6, True,  4),
                (80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, True,  4),
                (80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, True,  4),
                (80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, True,  4),
                (120, [3],          [1],    [1],    1, 6, True,  2),
                (120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, True,  2),
                (120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, True,  2),
                (120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, True,  2),
                (200, [3, 5, 7, 9], [1],    [1],    2, 6, True,  2),
                (200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, True,  2),
                (200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, True,  2),
                (200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, True,  2)
            ]
            first_channels_num = 24
            dropout = 0.25
            if mode == 'l':
                width_multiplier *= 1.3
        else:
            raise TypeError('Unsupported MixNet type')

        last_channels_num = 1536

        divisor = 8

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channels_num, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channels_num, momentum=BN_momentum, eps=BN_eps),
            nn.ReLU()
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for out_channels_num, kernel_sizes, exp_kernel_sizes, project_kernel_sizes, stride, exp_ratio, use_swish, se_ratio in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            feature_extraction_layers.append(Bottleneck(in_channels=input_channels_num, out_channels=output_channels_num, 
                                                        kernel_sizes=kernel_sizes, exp_kernel_sizes=exp_kernel_sizes, 
                                                        project_kernel_sizes=project_kernel_sizes, stride=stride, exp_ratio=exp_ratio,
                                                        use_swish=use_swish, se_ratio=se_ratio,
                                                        BN_momentum=BN_momentum, BN_eps=BN_eps))
            input_channels_num = output_channels_num
        
        # the last stage
        last_layer = nn.Sequential(
                nn.Conv2d(in_channels=input_channels_num, out_channels=last_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_channels_num, momentum=BN_momentum, eps=BN_eps),
                nn.ReLU()
            )
        feature_extraction_layers.append(last_layer)
        feature_extraction_layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*feature_extraction_layers)

        ########################################################################################################################
        # Classification part
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Sequential(),
            nn.Linear(last_channels_num, num_classes)
        )

        ########################################################################################################################
        # Initialize the weights
        self._initialize_weights(zero_gamma)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self, zero_gamma):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
	            if hasattr(m, 'lastBN'):
	                nn.init.constant_(m.lastBN.weight, 0.0)


if __name__ == '__main__':
    import argparse
    from torchsummaryX import summary
    parser = argparse.ArgumentParser(description='FLOPs and Params')
    parser.add_argument('--mode', type=str, default='s')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--wm', type=float, default=1.0)
    args = parser.parse_args()

    model = MixNet(mode=args.mode, num_classes=args.num_classes, input_size=args.input_size, width_multiplier=args.wm)
    model.eval()
    summary(model, torch.zeros((1, 3, args.input_size, args.input_size)))
    print('MixNet-%s-%.2f with input size %d and output %d classes' % (args.mode, args.wm, args.input_size, args.num_classes))
