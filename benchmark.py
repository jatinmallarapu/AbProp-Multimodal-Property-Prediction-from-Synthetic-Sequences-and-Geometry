"""Multimodal regression on artificial tokens and spatial coordinates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import torch
from torch import nn

TARGETS = ("token_signal", "geometry_signal", "interaction")


def dataset(seed):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(80), 16)
    prototypes = rng.integers(0, 8, (80, 16))
    tokens = prototypes[groups].copy()
    mask = rng.random(tokens.shape) < .16
    tokens[mask] = rng.integers(0, 8, mask.sum())
    points = rng.normal(size=(len(groups), 8, 3))
    distances = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)
    geometry = np.column_stack((distances.mean(1), distances.std(1), (distances < 2).mean(1)))
    weights = np.array([-2, -1, 0, 1, 2, 1, 0, -1])
    signal = weights[tokens[:, :8]].mean(1) - weights[tokens[:, 8:]].mean(1)
    spread = geometry[:, 0] - 2.2
    labels = np.column_stack((signal, spread, signal * spread))
    labels += rng.normal(0, .03, labels.shape)
    return tokens, geometry, labels, groups


def partition(groups, seed):
    order = np.random.default_rng(seed).permutation(np.unique(groups))
    return tuple(np.isin(groups, ids) for ids in np.split(order, [56, 68]))


def metrics(actual, predicted):
    return {
        name: {"rmse": float(np.sqrt(np.mean((a - p) ** 2))),
               "spearman": float(spearmanr(a, p).statistic) if np.ptp(p) else None}
        for name, a, p in zip(TARGETS, actual.T, predicted.T)
    }


def fit(features, labels, train, dev):
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    mean, scale = y[train].mean(0), y[train].std(0)
    y = (y - mean) / scale
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(),
                          nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    best_loss = float("inf")
    for _ in range(80):
        for batch in torch.where(torch.as_tensor(train))[0][torch.randperm(train.sum())].split(64):
            optimizer.zero_grad()
            loss = (model(x[batch]) - y[batch]).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            validation = (model(x[dev]) - y[dev]).square().mean().item()
            if validation < best_loss:
                best_loss = validation
                best = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    with torch.no_grad():
        return (model(x) * scale + mean).numpy(), {
            "weights": best, "target_mean": mean.tolist(), "target_scale": scale.tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    tokens, geo, labels, groups = dataset(args.seed)
    train, dev, test = partition(groups, args.seed)
    seq = np.eye(8)[tokens].reshape(len(tokens), -1)
    mean = geo[train].mean(0)
    scale = geo[train].std(0)
    geo = (geo - mean) / scale
    args.output.mkdir(parents=True, exist_ok=True)
    ridge = Ridge(alpha=1).fit(seq[train], labels[train])
    report = {"ridge": metrics(labels[test], ridge.predict(seq[test]))}
    for name, features in {"sequence": seq, "geometry": geo, "fusion": np.c_[seq, geo]}.items():
        predicted, weights = fit(features, labels, train, dev)
        report[name] = metrics(labels[test], predicted[test])
        if name == "fusion":
            torch.save({**weights, "geometry_mean": mean.tolist(),
                        "geometry_scale": scale.tolist()}, args.output / "fusion.pt")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for idx, ax in enumerate(axes):
        actual = labels[test, idx]
        estimate = predicted[test, idx]
        lo, hi = min(actual.min(), estimate.min()), max(actual.max(), estimate.max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")
        ax.scatter(actual, estimate, s=12, alpha=.55)
        ax.set(title=TARGETS[idx], xlabel="Actual", ylabel="Predicted")
    fig.savefig(args.output / "predictions.svg")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
