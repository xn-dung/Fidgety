import argparse
from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch

from experiment import summarize
from run import parse_folds


def parser_args():
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
    parser.add_argument("--gpu-workers", type=int, default=2)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def worker_command(args, fold, destination):
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("run.py")),
        "--data-path", args.data_path,
        "--output-dir", str(destination),
        "--folds", str(fold),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--patch-size", str(args.patch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--latent-dim", str(args.latent_dim),
        "--codebook-size", str(args.codebook_size),
        "--quantizers", str(args.quantizers),
        "--ngram-order", str(args.ngram_order),
        "--classifier-max-iter", str(args.classifier_max_iter),
        "--device", "cuda"
    ]
    if args.no_resume:
        command.append("--no-resume")
    return command


def complete_job(destination, fold):
    path = destination / "fold_results.csv"
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    actual = set(zip(frame["fold"], frame["classifier"]))
    return {(fold, "tfidf_logistic"), (fold, "tfidf_mlp")}.issubset(actual)


def merge_jobs(output, jobs, folds):
    frames = []
    for fold in folds:
        job = jobs / f"fold_{fold}"
        frame = pd.read_csv(job / "fold_results.csv")
        frames.append(frame[frame["fold"] == fold])
        source = job / f"fold_{fold}"
        destination = output / f"fold_{fold}"
        shutil.copytree(source, destination, dirs_exist_ok=True)
    results = pd.concat(frames, ignore_index=True).sort_values(["classifier", "fold"])
    summary = summarize(results)
    results.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    return results, summary


def main():
    args = parser_args()
    folds = parse_folds(args.folds)
    if args.gpu_workers < 1 or args.cpu_threads_per_worker < 1:
        raise ValueError("Worker counts must be positive")
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("No CUDA GPU is visible")
    worker_count = min(args.gpu_workers, available, len(folds))
    output = Path(args.output_dir)
    jobs = output / "_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    launch_config = vars(args).copy()
    launch_config["folds"] = list(folds)
    launch_config["visible_gpus"] = available
    launch_config["active_gpu_workers"] = worker_count
    config_path = output / "multi_gpu_config.json"
    if not args.no_resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != launch_config:
            raise ValueError("Existing output uses a different multi-GPU configuration")
    config_path.write_text(json.dumps(launch_config, indent=2), encoding="utf-8")
    pending = deque(fold for fold in folds if args.no_resume or not complete_job(jobs / f"fold_{fold}", fold))
    active = {}
    print(f"visible_gpus={available} gpu_workers={worker_count} folds={list(folds)}", flush=True)
    for gpu in range(worker_count):
        print(f"gpu={gpu} name={torch.cuda.get_device_name(gpu)}", flush=True)
    while pending or active:
        free = [gpu for gpu in range(worker_count) if gpu not in active]
        for gpu in free:
            if not pending:
                break
            fold = pending.popleft()
            destination = jobs / f"fold_{fold}"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["OMP_NUM_THREADS"] = str(args.cpu_threads_per_worker)
            environment["MKL_NUM_THREADS"] = str(args.cpu_threads_per_worker)
            process = subprocess.Popen(worker_command(args, fold, destination), env=environment)
            active[gpu] = (process, fold)
            print(f"gpu={gpu} started_fold={fold} pid={process.pid}", flush=True)
        if active:
            time.sleep(1)
        finished = []
        for gpu, (process, fold) in active.items():
            returncode = process.poll()
            if returncode is None:
                continue
            if returncode:
                for other, _ in active.values():
                    if other.poll() is None:
                        other.terminate()
                for other, _ in active.values():
                    other.wait()
                raise RuntimeError(f"Fold {fold} failed on GPU {gpu} with code {returncode}")
            print(f"gpu={gpu} completed_fold={fold}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
    _, summary = merge_jobs(output, jobs, folds)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
