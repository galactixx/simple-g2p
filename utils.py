import pickle
from pathlib import Path
from typing import Dict, List, TypeAlias
import os
import random

import torch
import numpy as np
from huggingface_hub import hf_hub_download

from constants import CMU_DICT_PATH

CMUDictType: TypeAlias = Dict[str, List[List[str]]]


def load_cmu_dict() -> CMUDictType:
    with open(CMU_DICT_PATH, "rb") as f:
        return pickle.load(f)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make CUDA algorithms deterministic where possible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Disable non-deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_model_checkpoint(filename: str) -> Path:
    checkpoint_path = hf_hub_download(
        repo_id="galactixx/cryogrid-boxes", filename=filename, token=False
    )
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path
