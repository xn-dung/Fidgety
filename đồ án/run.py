import argparse

from config import ExperimentConfig
from experiment import run_experiment


def parse_folds(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--quantizers", type=int, default=2)
    parser.add_argument("--ngram-order", type=int, default=2)
    parser.add_argument("--classifier-max-iter", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(data_path=args.data_path, output_dir=args.output_dir, folds=parse_folds(args.folds), seed=args.seed, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, patch_size=args.patch_size, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, codebook_size=args.codebook_size, quantizers=args.quantizers, ngram_order=args.ngram_order, classifier_max_iter=args.classifier_max_iter, device=args.device, resume=not args.no_resume)
    _, summary = run_experiment(config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
