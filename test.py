import argparse

import torch

from constants import SEED
from dataset import create_dataloaders
from evaluation import beam_search_evaluation, greedy_decode_evaluation
from models import MODELS
from preprocessing import build_ref_map, parse_cmu_dict, split_and_generate_pairs
from utils import get_model_checkpoint, load_cmu_dict, seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    seed_everything(seed=SEED)
    parser = argparse.ArgumentParser(
        description="Test a G2P model using different model types."
    )
    parser.add_argument(
        "--decode",
        choices=["greedy", "beam"],
        required=True,
        help="The decoding strategy to use.",
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
    filename, _, model_init = model_info

    model_path = get_model_checkpoint(filename=filename)

    cmu_dict = load_cmu_dict()
    graphemes, config = parse_cmu_dict(cmu_dict=cmu_dict)
    train, val, test = split_and_generate_pairs(graphemes, config)
    ref_map = build_ref_map(train, val, test, graphemes)

    _, _, test_loader = create_dataloaders(config, train, val, test)

    model = model_init(config=config)

    weights = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    if args.decode == "greedy":
        seq_acc = greedy_decode_evaluation(model, test_loader, config, ref_map)
    else:
        seq_acc = beam_search_evaluation(model, test_loader, config, ref_map)

    print(f"{args.decode.capitalize()} sequence accuracy: {seq_acc:.3f}")
