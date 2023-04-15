import torch
import torch.nn as nn
from torchvision.models import resnet50



class ResNet(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()

        self.output_size = output_size

        self.net = resnet50(num_classes=output_size)
        self.reset_parameters()

    def _init_weights(self, m):
        # raise RuntimeError
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)


    @torch.no_grad()
    def net_forward(self, x):

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, returnt="out") -> torch.Tensor:
        return self.net(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, returnt="features")

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads
