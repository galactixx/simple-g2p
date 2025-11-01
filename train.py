import argparse
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
from evaluation import greedy_decode_evaluation
from models import MODELS
from preprocessing import (
    TokenConfig,
    build_ref_map,
    parse_cmu_dict,
    split_and_generate_pairs,
)
from utils import load_cmu_dict, seed_everything

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def token_level_evaluate(
    model: torch.nn.Module,
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
    seed_everything(seed=SEED)
    parser = argparse.ArgumentParser(
        description="Traing a G2P model using different model types."
    )
    parser.add_argument(
        "--model",
        choices=["lstm"],
        required=True,
        help="The g2p model to use.",
    )
    args = parser.parse_args()

    model_info = MODELS.get(args.model, None)
    assert model_info is not None
    _, _, model_init = model_info

    cmu_dict = load_cmu_dict()
    graphemes, config = parse_cmu_dict(cmu_dict=cmu_dict)
    train, val, test = split_and_generate_pairs(graphemes, config)
    ref_map = build_ref_map(train, val, test, graphemes)

    train_loader, val_loader, _ = create_dataloaders(config, train, val, test)

    DECODE_PAD_ID = config.decode_char_to_id[PAD]

    model = model_init(config=config)
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
        val_seq_acc = greedy_decode_evaluation(model, val_loader, config, ref_map, ema)
        scheduler.step(val_loss)

        print(
            f"Current Prob: {p}.. "
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
