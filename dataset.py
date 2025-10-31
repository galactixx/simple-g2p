import random
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from constants import PAD, SEED
from preprocessing import PhonemePair, TokenConfig


class CMUPhonemeDataset(Dataset):
    def __init__(self, pairs: List[PhonemePair]) -> None:
        super().__init__()
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        grapheme = torch.tensor(pair.grapheme)
        phoneme = torch.tensor(pair.phoneme)
        return pair.index, grapheme, phoneme


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def get_padded_seq(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    return pad_sequence(seqs, batch_first=True, padding_value=pad_id)


def get_lengths(seqs: List[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([len(seq) for seq in seqs], dtype=torch.long)


def _collate_fn(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    encode_pad_id: int,
    decode_pad_id: int,
):
    indices, char_seqs, phoneme_seqs = zip(*batch)

    enc_padded = get_padded_seq(seqs=char_seqs, pad_id=encode_pad_id)
    dec_padded = get_padded_seq(seqs=phoneme_seqs, pad_id=decode_pad_id)

    enc_lens = get_lengths(seqs=char_seqs)
    indices = torch.tensor(indices)
    return indices, enc_padded, enc_lens, dec_padded


def create_dataloaders(
    config: TokenConfig,
    train_pairs: List[PhonemePair],
    val_pairs: List[PhonemePair],
    test_pairs: List[PhonemePair],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    DECODE_PAD_ID = config.decode_char_to_id[PAD]
    ENCODE_PAD_ID = config.encode_char_to_id[PAD]

    train_dataset = CMUPhonemeDataset(pairs=train_pairs)
    val_dataset = CMUPhonemeDataset(pairs=val_pairs)
    test_dataset = CMUPhonemeDataset(pairs=test_pairs)

    g = torch.Generator()
    g.manual_seed(SEED)

    collate_fn = partial(
        _collate_fn,
        encode_pad_id=ENCODE_PAD_ID,
        decode_pad_id=DECODE_PAD_ID,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
        batch_size=64,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=32,
        num_workers=2,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=32,
        num_workers=2,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader
