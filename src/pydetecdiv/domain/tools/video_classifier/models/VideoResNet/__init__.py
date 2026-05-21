import sys
from typing import Union, Sequence, Callable

from torch import nn, Tensor
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18, r2plus1d_18, R2Plus1D_18_Weights, VideoResNet
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, Bottleneck, Conv3DSimple, Conv3DNoTemporal


class R3D_18(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(R3D_18, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.r3d_18 = r3d_18(weights=R3D_18_Weights.DEFAULT)

        self.r3d_18.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                       nn.Linear(in_features=512, out_features=n_classes, bias=True)
                                       )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.r3d_18(x)
        return x


class MC3_18(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(MC3_18, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.mc3_18 = mc3_18(weights=MC3_18_Weights.DEFAULT)

        self.mc3_18.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                       nn.Linear(in_features=512, out_features=n_classes, bias=True)
                                       )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.mc3_18(x)
        return x


class R2Plus1d_18(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(R2Plus1d_18, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.r2plus1d_18 = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        self.linear = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        self.r2plus1d_18.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                            self.linear
                                            )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.r2plus1d_18(x)
        return x


class CustomVideoResnet(nn.Module):
    def __init__(
            self,
            block: type[Union[BasicBlock, Bottleneck]],
            conv_makers: Sequence[type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
            layers: list[int],
            strides: list[int],
            stem: Callable[..., nn.Module],
            num_classes: int = 400,
            zero_init_residual: bool = False,
            **kwargs
            ) -> None:

        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        # change that as it is only a workaround, temporarily creating a model just for the purpose of creating all required attributes
        # super().__init__(BasicBlock, [Conv2Plus1D] * 4, [2, 2, 2, 2], R2Plus1dStem, **kwargs)
        super().__init__()
        self.inplanes = 64
        print(layers, file=sys.stderr)

        self.stem = stem()

        self.layers = nn.Sequential(*[self._make_layer(block, conv_makers[i], 64 * (i + 1), layer, stride=strides[i]) for i, layer in enumerate(layers)])
        # self.layers = []
        #
        # for i, layer in enumerate(layers):
        #     print(f'Layer {i}')
        #     self.layers.append(self._make_layer(block, conv_makers[i], 64 * (i + 1), layer, stride=strides[i]))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(64 * len(layers) * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(
            self,
            block: type[Union[BasicBlock, Bottleneck]],
            conv_builder: type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]],
            planes: int,
            blocks: int,
            stride: int = 1,
            ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                    )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


class CustomR2Plus_1D(nn.Module):
    def __init__(self, n_classes=6, dropout=0.2, layers=None, strides=None, **kwargs):
        super(CustomR2Plus_1D, self).__init__()
        self.expected_shape = ('Batch', 4, 3, 224, 224)

        if layers is None:
            layers = [2, 2, 2, 2]

        blocks = len(layers)
        if strides is None:
            strides = [1] + [2] * (blocks - 1)

        self.model = CustomVideoResnet(BasicBlock, [Conv2Plus1D] * blocks, layers, strides, R2Plus1dStem, n_classes,
                                       True, **kwargs)

        self.model.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                      self.model.fc
                                      )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        return x
