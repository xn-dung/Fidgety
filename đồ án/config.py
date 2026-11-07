from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_path: str
    output_dir: str
    folds: tuple[int, ...] = tuple(range(1, 10))
    seed: int = 42
    epochs: int = 80
    patience: int = 10
    batch_size: int = 64
    patch_size: int = 10
    hidden_dim: int = 256
    latent_dim: int = 64
    codebook_size: int = 64
    quantizers: int = 2
    commitment_weight: float = 0.25
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    ngram_order: int = 2
    mlp_hidden_dim: int = 128
    classifier_max_iter: int = 500
    device: str = "auto"
    resume: bool = True

    def validate(self):
        if not self.folds or any(fold < 1 or fold > 9 for fold in self.folds):
            raise ValueError("folds must contain values from 1 to 9")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if 250 % self.patch_size:
            raise ValueError("patch_size must divide 250")
        for value in [self.epochs, self.patience, self.batch_size, self.hidden_dim, self.latent_dim, self.codebook_size, self.quantizers, self.ngram_order]:
            if value < 1:
                raise ValueError("positive configuration values are required")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        result = asdict(self)
        result["folds"] = list(self.folds)
        return result
