from functools import partial

import torch
from torchvision.models import get_model


def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)

    return x


def main():
    m = get_model("resnet18")
    m._forward_impl = partial(_forward_impl, m)

    x = torch.randn(1, 3, 224, 224)

    y = m(x)

    print(y.shape)


if __name__ == "__main__":
    main()
