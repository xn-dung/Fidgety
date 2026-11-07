import json
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from config import ExperimentConfig
from data import TextStore, load_video_fold
from features import codebook_statistics, fit_tfidf
from model import RVQVAE, encode_tokens, train_rvqvae


PAPER_VIDEO_BALANCED_ACCURACY = 0.9066064666666666


def save_object(value, path):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {"sensitivity": sensitivity, "specificity": specificity, "balanced_accuracy": (sensitivity + specificity) / 2, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classifiers(config, fold):
    return {
        "tfidf_logistic": LogisticRegression(max_iter=config.classifier_max_iter, class_weight="balanced", random_state=config.seed + fold),
        "tfidf_mlp": MLPClassifier(hidden_layer_sizes=(config.mlp_hidden_dim,), max_iter=config.classifier_max_iter, early_stopping=True, validation_fraction=0.125, n_iter_no_change=20, batch_size=64, random_state=config.seed + fold)
    }


def summarize(frame):
    rows = []
    for name, group in frame.groupby("classifier"):
        row = {"classifier": name, "n_folds": len(group)}
        for metric in ["sensitivity", "specificity", "balanced_accuracy"]:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            margin = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            row[metric + "_mean"] = mean
            row[metric + "_ci_low"] = mean - margin
            row[metric + "_ci_high"] = mean + margin
        row["paper_video_balanced_accuracy"] = PAPER_VIDEO_BALANCED_ACCURACY
        row["balanced_accuracy_delta"] = row["balanced_accuracy_mean"] - PAPER_VIDEO_BALANCED_ACCURACY
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig):
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    config_path = output / "config.json"
    current_config = config.to_dict()
    if config.resume and config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != current_config:
        raise ValueError("Existing output uses a different configuration")
    config_path.write_text(json.dumps(current_config, ensure_ascii=False, indent=2), encoding="utf-8")
    store = TextStore(config.data_path)
    result_path = output / "fold_results.csv"
    existing = pd.read_csv(result_path) if config.resume and result_path.is_file() else pd.DataFrame()
    rows = existing.to_dict("records")
    complete = set(zip(existing.get("fold", []), existing.get("classifier", []))) if len(existing) else set()
    print(f"device={device} folds={list(config.folds)}", flush=True)
    for fold in config.folds:
        expected = {(fold, "tfidf_logistic"), (fold, "tfidf_mlp")}
        if config.resume and expected.issubset(complete):
            print(f"fold={fold} skipped", flush=True)
            continue
        rows = [row for row in rows if int(row["fold"]) != fold]
        seed_everything(config.seed + fold)
        print(f"fold={fold} loading", flush=True)
        data = load_video_fold(store, fold)
        model = RVQVAE(patch_size=config.patch_size, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim, codebook_size=config.codebook_size, quantizers=config.quantizers, commitment_weight=config.commitment_weight)
        history = train_rvqvae(model, data.x_fit, data.x_val, device, config.epochs, config.patience, config.batch_size, config.learning_rate, config.weight_decay)
        train_tokens = encode_tokens(model, data.x_train, device, config.batch_size)
        test_tokens = encode_tokens(model, data.x_test, device, config.batch_size)
        vectorizer, x_train, x_test = fit_tfidf(train_tokens, test_tokens, config.ngram_order)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config.to_dict()}, fold_dir / "rvqvae.pt")
        save_object(vectorizer, fold_dir / "tfidf.pkl")
        pd.DataFrame(history).to_csv(fold_dir / "rvqvae_history.csv", index=False)
        (fold_dir / "codebook_statistics.json").write_text(json.dumps(codebook_statistics(train_tokens, config.codebook_size), indent=2), encoding="utf-8")
        for name, classifier in classifiers(config, fold).items():
            classifier.fit(x_train, data.y_train)
            prediction = classifier.predict(x_test)
            metrics = classification_metrics(data.y_test, prediction)
            row = {"fold": fold, "classifier": name, "train_samples": len(data.y_train), "test_samples": len(data.y_test), "vocabulary_size": len(vectorizer.vocabulary_), **metrics}
            rows.append(row)
            save_object(classifier, fold_dir / f"{name}.pkl")
            pd.DataFrame({"y_true": data.y_test, "y_pred": prediction}).to_csv(fold_dir / f"{name}_predictions.csv", index=False)
            print(f"fold={fold} classifier={name} balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
        frame = pd.DataFrame(rows).sort_values(["classifier", "fold"])
        frame.to_csv(result_path, index=False)
        summarize(frame).to_csv(output / "summary.csv", index=False)
    frame = pd.read_csv(result_path)
    summary = summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    return frame, summary
