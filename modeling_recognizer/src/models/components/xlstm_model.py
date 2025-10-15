import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig


class xLSTMCompatibleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
        track_state: bool = False,  # сохранять hx
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.track_state = track_state

        self.num_directions = 2 if bidirectional else 1

        # конфигурация для xLSTM
        self.fwd_cfg = xLSTMBlockStackConfig(
            context_length=None,
            embedding_dim=input_size,
            num_blocks=num_layers,
            dropout=dropout,
            slstm_at=[i for i in range(num_layers)],
        )
        self.fwd = xLSTMBlockStack(self.fwd_cfg)

        if self.bidirectional:
            self.bwd_cfg = xLSTMBlockStackConfig(
                context_length=None,
                embedding_dim=input_size,
                num_blocks=num_layers,
                dropout=dropout,
                slstm_at=[i for i in range(num_layers)],
            )
            self.bwd = xLSTMBlockStack(self.bwd_cfg)

        self.out_proj = nn.Linear(
            input_size * self.num_directions, hidden_size
        )

        self._last_hn = None
        self._last_cn = None

    def forward(self, x, hx=None):
        # x: (B, T, C) if batch_first=True
        if not self.batch_first:
            x = x.transpose(0, 1)  # (T, B, C) → (B, T, C)

        # forward
        out_fwd = self.fwd(x)

        if self.bidirectional:
            out_bwd = self.bwd(torch.flip(x, dims=[1]))  # reverse in time
            out_bwd = torch.flip(out_bwd, dims=[1])      # flip back
            out = torch.cat([out_fwd, out_bwd], dim=-1)
        else:
            out = out_fwd

        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        batch_size = x.size(0)
        device = x.device

        hn = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        cn = torch.zeros_like(hn)

        if self.track_state:
            self._last_hn = hn
            self._last_cn = cn

        return out, (hn, cn)

    def get_last_state(self):
        return self._last_hn, self._last_cn