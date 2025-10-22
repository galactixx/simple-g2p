import pickle
from pathlib import Path
from typing import Dict, List, TypeAlias

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
