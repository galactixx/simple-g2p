import argparse

import matplotlib.pyplot as plt
import pandas as pd

from models import MODELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and save loss graphs from training runs."
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
    _, loss_filename, _ = model_info

    data = pd.read_csv(f"./losses/{loss_filename}")

    # plot the training and validation losses against sequence
    # level accuracy
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        data["epoch"],
        data["train_loss"],
        label="Train Loss",
        color="blue",
        linestyle="-",
        linewidth=1.8,
    )
    ax1.plot(
        data["epoch"],
        data["val_loss"],
        label="Val Loss",
        color="red",
        linestyle="-",
        linewidth=1.8,
    )
    ax2.plot(
        data["epoch"],
        data["val_seq_acc"],
        color="green",
        label="Val Accuracy",
        linestyle=":",
        linewidth=2,
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Sequence Accuracy")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Training & Validation Loss vs Accuracy")

    plt.tight_layout()
    plt.savefig(
        f"./graphs/{args.model}-loss-vs-seq-accuracy.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # plot the training and validation losses against teacher
    # forcing probability
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        data["epoch"],
        data["train_loss"],
        label="Train Loss",
        color="blue",
        linestyle="-",
        linewidth=1.8,
    )
    ax1.plot(
        data["epoch"],
        data["val_loss"],
        label="Val Loss",
        color="red",
        linestyle="-",
        linewidth=1.8,
    )
    ax2.plot(
        data["epoch"],
        data["prob"],
        color="purple",
        label="Teacher Forcing Prob",
        linestyle=":",
        linewidth=2,
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Teacher Forcing Probability")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Teacher Forcing Probability vs Loss")

    plt.tight_layout()
    plt.savefig(
        f"./graphs/{args.model}-loss-vs-teacher-force.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
