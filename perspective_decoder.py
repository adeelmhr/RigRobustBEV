import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDown(nn.Module):
    def __init__(self, inC: int, outC: int, use_bn: bool = True, activation: str = "relu"):
        super().__init__()
        layers = [nn.Conv2d(inC, outC, kernel_size=3, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(outC))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        else:
            layers.append(nn.Identity())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownsampleMapper(nn.Module):
    """
    Simple CNN to map perspective features [B, C, Hp, Wp] down to neck resolution [B, C, Hp/8, Wp/8].
    Default: three stride-2 conv blocks keep channels constant.
    """

    def __init__(self, inC: int = 128, outC: int = 128, num_down: int = 3, use_bn: bool = True, activation: str = "relu"):
        super().__init__()
        assert num_down >= 1
        downs = []
        c_in = inC
        for _ in range(num_down):
            downs.append(ConvDown(c_in, outC, use_bn=use_bn, activation=activation))
            c_in = outC
        self.downs = nn.Sequential(*downs)
        # Light refinement at target scale
        act = nn.ReLU(inplace=True) if activation == "relu" else (nn.GELU() if activation == "gelu" else (nn.SiLU(inplace=True) if activation == "silu" else nn.Identity()))
        self.refine = nn.Sequential(
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(outC) if use_bn else nn.Identity(),
            act,
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.downs(x)
        y = self.refine(y)
        return y
