from typing import Callable, Dict, Tuple, Type, TypeAlias

import torch

from config import get_lstm_modules
from g2p import G2PLSTM
from preprocessing import TokenConfig

ModelInit: TypeAlias = Callable[[TokenConfig], torch.nn.Module]


def lstm_init(config: TokenConfig) -> torch.nn.Module:
    enc_module, dec_module = get_lstm_modules(config=config)
    return G2PLSTM(enc=enc_module, dec=dec_module)


MODELS: Dict[str, Tuple[str, str, ModelInit]] = {
    "lstm": ("g2p-lstm.bin", "lstm.csv", lstm_init)
}
