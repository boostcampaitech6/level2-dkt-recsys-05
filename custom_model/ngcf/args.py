import argparse
import enum
import yaml
from pydantic import BaseModel


class Mode(str, enum.Enum):
    Train = "Train"
    Inference = "Inference"


class Args(BaseModel):
    seed: int = 42
    use_cuda_if_available: bool = True
    data_dir: str = "/opt/ml/data"
    output_dir: str = "./outputs/"
    node_dropout: float = 0.2
    mess_dropout: float = 0.2
    embedding_dim: int = 64
    layers: list = [64, 64]
    batch_size: int = 64
    alpha: float | None = None
    n_epochs: int = 20
    lr: float = 0.001
    model_dir: str = "./models/"
    model_name: str = "best_model.pt"
    mode: Mode = Mode.Train


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="YAML configuration file")

    with open(parser.parse_known_args()[0].config, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**args)

    args = parser.parse_args()

    return Args(**args.__dict__)
