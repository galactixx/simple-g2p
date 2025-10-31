from typing import Tuple

from g2p import LSTMModule
from preprocessing import TokenConfig


def get_lstm_modules(config: TokenConfig) -> Tuple[LSTMModule, LSTMModule]:
    enc_vocab_dim = len(config.encode_vocab)
    dec_vocab_dim = len(config.decode_vocab)

    DECODE_PAD_ID = config.decode_char_to_id[PAD]
    ENCODE_PAD_ID = config.encode_char_to_id[PAD]

    enc_module = LSTMModule(
        vocab=enc_vocab_dim,
        embed=256,
        hidden=512,
        layers=2,
        dropout=0.35,
        pad_id=ENCODE_PAD_ID,
    )
    dec_module = LSTMModule(
        vocab=dec_vocab_dim,
        embed=256,
        hidden=512,
        layers=2,
        dropout=0.35,
        pad_id=DECODE_PAD_ID,
    )
    return enc_module, dec_module
