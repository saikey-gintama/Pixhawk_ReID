"""
tcn.py
=======
locuslab/TCN (https://github.com/locuslab/TCN) 의 TCN/tcn.py 에 정의된
Chomp1d / TemporalBlock / TemporalConvNet 아키텍처를 그대로 옮긴 것 (MIT
라이선스 원 저장소를 클론하지 않고 아키텍처 정의만 복사 -- 벤치마크
태스크 코드는 가져오지 않음). 분류 head(TCNClassifier)만 이 프로젝트에서
새로 추가.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """causal padding으로 conv 출력 오른쪽에 남는 미래 시점 padding을 잘라낸다."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNClassifier(nn.Module):
    """TemporalConvNet + 마지막 시점(causal이므로 '현재') 은닉상태에 선형 분류 head.
    입력 x: (batch, seq_len) 단일 피처(z-score) -> 채널 1로 unsqueeze."""

    def __init__(self, input_size: int = 1, num_channels=(16, 16, 16), kernel_size: int = 3,
                 dropout: float = 0.2, n_classes: int = 3):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, list(num_channels), kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (batch, 1, seq_len)
        y = self.tcn(x)              # (batch, channels, seq_len)
        y = y[:, :, -1]               # 마지막(=현재) 시점
        return self.fc(y)
