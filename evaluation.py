from typing import Optional

import torch
import torch.nn.functional as F
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


def beam_search(
    model: torch.nn.Module,
    grapheme: torch.Tensor,
    grapheme_lens: torch.Tensor,
    phoneme: torch.Tensor,
    config: TokenConfig,
) -> torch.Tensor:
    BOS_ID = config.decode_char_to_id[BOS]
    EOS_ID = config.decode_char_to_id[EOS]
    DECODE_PAD_ID = config.decode_char_to_id[PAD]

    K = 5
    done_beams = []
    start = torch.tensor([BOS_ID], dtype=torch.long, device=device)
    cur_beams = [(start, 0, False)]
    grapheme = grapheme.unsqueeze(0)
    grapheme_lens = grapheme_lens.unsqueeze(0)
    is_done = False
    length = 1
    max_done_score = float("-inf")

    while not is_done and length < phoneme.size(0):
        new_beams = []
        done_beams = done_beams + [b for b in cur_beams if b[2]]
        if done_beams:
            max_done_score = max(max_done_score, max(b[1] for b in done_beams))

        cur_beams = [b for b in cur_beams if not b[2]]
        if not cur_beams:
            cur_beams.extend(done_beams)
            break

        batch_seqs = torch.nn.utils.rnn.pad_sequence(
            [seq for seq, _, _ in cur_beams],
            batch_first=True,
            padding_value=DECODE_PAD_ID,
        )

        grapheme_batch = grapheme.expand(len(cur_beams), -1)
        grapheme_lens_batch = grapheme_lens.expand(len(cur_beams))

        logits = model(grapheme_batch, grapheme_lens_batch, batch_seqs, 0)
        logits = logits[:, -1, :]
        logits_log = F.log_softmax(logits, dim=-1)

        for i, (beam, score, _) in enumerate(cur_beams):
            values, indices = logits_log[i].topk(K, dim=-1)

            for value, idx in zip(values, indices):
                token = idx.view(1)
                new_score = score + value.item()
                new_beam = torch.cat([beam, token], dim=-1)

                triplet = (new_beam, new_score, idx == EOS_ID)
                new_beams.append(triplet)

        is_done = all(b[1] < max_done_score for b in new_beams)

        cur_beams = new_beams + done_beams
        cur_beams = sorted(cur_beams, key=lambda x: x[1] / len(x[0]), reverse=True)
        cur_beams = cur_beams[:K]

        length += 1

    cur_beams.sort(key=lambda x: (x[2], x[1] / len(x[0])), reverse=True)
    return cur_beams[0][0]


def beam_search_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    config: TokenConfig,
    ref_map: RefMap,
) -> float:
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for idxs, graphemes, graphemes_lens, phonemes in tqdm(loader):
            idxs, graphemes, phonemes = (
                idxs.to(device),
                graphemes.to(device),
                phonemes.to(device),
            )
            batch_size = graphemes.size(0)
            decoded_seqs = []

            for i in range(batch_size):
                decoded_seq = beam_search(
                    model, graphemes[i], graphemes_lens[i], phonemes[i], config
                )
                decoded_seqs.append(decoded_seq)

            for i in range(batch_size):
                pred_seq = _mask_out(decoded_seqs[i], config)
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
    return seq_acc
