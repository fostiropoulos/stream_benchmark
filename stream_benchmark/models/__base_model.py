from abc import ABC, abstractmethod
from functools import cached_property
import torch.nn as nn
from torch.optim import SGD, AdamW
import torch


class BaseModel(nn.Module, ABC):
    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        lr: float,
    ) -> None:
        super(BaseModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.lr = lr
        self.optimizer = AdamW(self.net.parameters(), lr=self.lr)

    @cached_property
    def device(self):
        return next(iter(self.net.parameters())).device

    def to(self, device):
        super().to(device)
        if "device" in self.__dict__:
            del self.__dict__["device"]
        if hasattr(self, "buffer"):
            self.buffer.output_device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier of the method"""
        pass

    @property
    @abstractmethod
    def link(self) -> str:
        """link to the page of the method (not the pdf, but i.e. the arxiv page or abstract page)"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human description of the method. i.e. `Stochastic gradient descent baseline without continual learning.`"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_optim(self):
        self.optimizer = SGD(self.net.parameters(), lr=self.lr)

    @abstractmethod
    def begin_task(self):
        pass

    @abstractmethod
    def observe(
        self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor
    ) -> float:
        pass

    @abstractmethod
    def end_task(self, *args, **kwargs):
        pass
