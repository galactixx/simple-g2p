import warnings

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from constants import BOS, EOS, PAD
from dataset import create_dataloaders
from g2p import G2PLSTM, LSTMModule
from preprocessing import (
    TokenConfig,
    build_ref_map,
    parse_cmu_dict,
    split_and_generate_pairs,
)
from utils import load_cmu_dict

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_out(x: torch.Tensor, config: TokenConfig) -> torch.Tensor:
    PAD_ID = config.decode_char_to_id[PAD]
    EOS_ID = config.decode_char_to_id[EOS]
    BOS_ID = config.decode_char_to_id[BOS]

    mask = (x != EOS_ID) & (x != BOS_ID) & (x != PAD_ID)
    return x[mask]


def seq_compare(ref: torch.Tensor, pred: torch.Tensor) -> bool:
    if ref.size(0) != pred.size(0):
        return False
    return torch.equal(ref, pred)


def seq_ref_compare(ref: torch.Tensor, pred: torch.Tensor, config: TokenConfig) -> bool:
    ref = mask_out(ref, config)
    return seq_compare(ref=ref, pred=pred)


def seq_level_evaluate(
    model: G2PLSTM,
    loader: DataLoader,
    ema: ExponentialMovingAverage,
    config: TokenConfig,
) -> float:
    model.eval()
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
                pred_seq = mask_out(dec_in[i], config)
                target_seq = mask_out(phonemes[i], config)

                index = idxs[i].item()
                if index in ref_map:
                    multi_refs = ref_map[index]
                    correct += any(
                        seq_ref_compare(
                            torch.tensor(ref, device=device), pred_seq, config
                        )
                        for ref in multi_refs
                    )
                else:
                    correct += seq_compare(pred_seq, target_seq)

            total += graphemes.size(0)

        seq_acc = correct / total
    ema.restore()
    return seq_acc


def token_level_evaluate(
    model: G2PLSTM,
    criterion: CrossEntropyLoss,
    ema: ExponentialMovingAverage,
    loader: DataLoader,
) -> float:
    model.eval()
    ema.store()
    ema.copy_to()

    running_loss = 0
    with torch.no_grad():
        for _, graphemes, graphemes_lens, phonemes in tqdm(loader):
            graphemes, phonemes = graphemes.to(device), phonemes.to(device)

            dec_in = phonemes[:, :-1]
            dec_target = phonemes[:, 1:]

            logits = model(graphemes, graphemes_lens, dec_in, 0).transpose(1, 2)
            loss = criterion(logits, dec_target)

            running_loss += loss.item() * graphemes.size(0)

    eval_loss = running_loss / len(loader.dataset)
    ema.restore()
    return eval_loss


if __name__ == "__main__":
    cmu_dict = load_cmu_dict()
    graphemes, config = parse_cmu_dict(cmu_dict=cmu_dict)
    train, val, test = split_and_generate_pairs(graphemes, config)
    ref_map = build_ref_map(train, val, test, graphemes)

    train_loader, val_loader, test_loader = create_dataloaders(config, train, val, test)

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

    model = G2PLSTM(enc=enc_module, dec=dec_module)
    model.to(device)

    SCHE_PATIENCE = 3

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=SCHE_PATIENCE
    )
    criterion = CrossEntropyLoss(label_smoothing=0.10, ignore_index=DECODE_PAD_ID)

    EPOCHS = 100
    PATIENCE = 7

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    scaler = GradScaler()
    best_loss, no_improve = float("inf"), 0

    p = 1.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for _, graphemes, graphemes_lens, phonemes in tqdm(train_loader):
            graphemes, phonemes = graphemes.to(device), phonemes.to(device)

            dec_in = phonemes[:, :-1]
            dec_target = phonemes[:, 1:]

            optimizer.zero_grad()

            with autocast():
                logits = model(graphemes, graphemes_lens, dec_in, p)
                loss = criterion(logits.transpose(1, 2), dec_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            running_loss += loss.item() * graphemes.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = token_level_evaluate(model, criterion, ema, val_loader)
        val_seq_acc = seq_level_evaluate(model, val_loader, ema, config)
        scheduler.step(val_loss)

        print(
            f"Current Prob: {max(0.20, 1.0 - step / 80_000)}.. "
            f"Epoch {epoch+1}/{EPOCHS}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Val loss: {val_loss:.3f}.. "
            f"Val sequence accuracy: {val_seq_acc:.3f}.."
        )

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            ema.copy_to()
            torch.save(model.state_dict(), "g2p-lstm.bin")
            ema.restore()
        else:
            no_improve += 1
            if no_improve >= SCHE_PATIENCE:
                p *= 0.8

            if no_improve >= PATIENCE:
                break
