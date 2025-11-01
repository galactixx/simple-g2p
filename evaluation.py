from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from constants import BOS, EOS, PAD
from preprocessing import RefMap, TokenConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mask_out(x: torch.Tensor, config: TokenConfig) -> torch.Tensor:
    PAD_ID = config.decode_char_to_id[PAD]
    EOS_ID = config.decode_char_to_id[EOS]
    BOS_ID = config.decode_char_to_id[BOS]

    mask = (x != EOS_ID) & (x != BOS_ID) & (x != PAD_ID)
    return x[mask]


def _seq_compare(ref: torch.Tensor, pred: torch.Tensor) -> bool:
    if ref.size(0) != pred.size(0):
        return False
    return torch.equal(ref, pred)


def _seq_ref_compare(
    ref: torch.Tensor, pred: torch.Tensor, config: TokenConfig
) -> bool:
    ref = _mask_out(ref, config)
    return _seq_compare(ref=ref, pred=pred)


def greedy_decode_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    config: TokenConfig,
    ref_map: RefMap,
    ema: Optional[ExponentialMovingAverage] = None,
) -> float:
    model.eval()

    if ema is not None:
        ema.store()
        ema.copy_to()
    EOS_ID = config.decode_char_to_id[EOS]
    BOS_ID = config.decode_char_to_id[BOS]

    with torch.no_grad():
        correct = total = 0
        for idxs, graphemes, graphemes_lens, phonemes in tqdm(loader):
            idxs, graphemes, phonemes = (
                idxs.to(device),
                graphemes.to(device),
                phonemes.to(device),
            )

            batch_size = graphemes.size(0)
            dec_in = torch.full(
                (batch_size, 1), BOS_ID, dtype=torch.long, device=device
            )

            eos_mask = torch.full((batch_size,), False, dtype=torch.bool, device=device)

            max_length = phonemes.size(1)
            done = 0
            while done < batch_size and dec_in.size(1) < max_length:
                logits = model(graphemes, graphemes_lens, dec_in, 0)
                pred = logits[:, -1, :].argmax(dim=1)
                pred[eos_mask] = EOS_ID

                dec_in = torch.cat([dec_in, pred.unsqueeze(1)], dim=1)
                eos_mask = eos_mask | (pred == EOS_ID)
                done = eos_mask.sum().item()

            for i in range(batch_size):
                pred_seq = _mask_out(dec_in[i], config)
                target_seq = _mask_out(phonemes[i], config)

                index = idxs[i].item()
                if index in ref_map:
                    multi_refs = ref_map[index]
                    correct += any(
                        _seq_ref_compare(
                            torch.tensor(ref, device=device), pred_seq, config
                        )
                        for ref in multi_refs
                    )
                else:
                    correct += _seq_compare(pred_seq, target_seq)

            total += graphemes.size(0)

        seq_acc = correct / total

    if ema is not None:
        ema.restore()
    return seq_acc
