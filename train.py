"""Train the JSCC codec, importance evaluator, or CSI predictor."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from channel import transmit
from data import ROOT, load_csi, load_images, load_weights
from models import ChannelPredictor, Codec, WNet, distillation_targets, normalize_features


class PredictionWindows(Dataset):
    def __init__(self, trace, limit):
        self.trace = trace
        available = len(trace) - ChannelPredictor.history
        self.count = available if limit is None else limit
        if not 1 <= self.count <= available:
            raise ValueError(f"Need between 1 and {available} prediction windows")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        end = index + ChannelPredictor.history
        return self.trace[index:end], self.trace[end]


def prepare_training(args, device):
    if args.stage == "predictor":
        dataset = PredictionWindows(load_csi(args.csi), args.limit)
        return ChannelPredictor(), dataset, None

    images = load_images(args.data_dir, train=True, limit=args.limit)
    if args.stage == "codec":
        trace = load_csi(args.csi)
        if len(trace) < len(images) * (Codec.channels // ChannelPredictor.antennas):
            raise ValueError("Codec training needs 12 CSI samples per training image")
        return Codec(), images, trace

    # The archived teacher used batch statistics during offline label generation.
    teacher = load_weights(Codec(), args.codec).to(device).train().requires_grad_(False)
    feature_batches, label_batches = [], []
    for batch, _ in DataLoader(images, batch_size=args.batch_size):
        features, scores = distillation_targets(teacher, batch.to(device))
        feature_batches.append(features.cpu())
        label_batches.append(scores.cpu())
    dataset = TensorDataset(torch.cat(feature_batches), torch.cat(label_batches))
    return WNet(), dataset, None


def train(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, dataset, trace = prepare_training(args, device)
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    default_lr = 1e-3 if args.stage == "predictor" else 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or default_lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        total_loss = 0.0
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            count = len(inputs)
            if args.stage == "codec":
                slots = Codec.channels // ChannelPredictor.antennas
                actual = trace[seen * slots:(seen + count) * slots].reshape(count, slots, 2, 2)
                features = normalize_features(model.encoder(inputs))
                received = transmit(features, actual.to(device), args.snr, "power_control")
                loss = F.mse_loss(model.decoder(received), inputs)
            elif args.stage == "importance":
                loss = F.mse_loss(model(inputs, args.snr), targets.to(device))
            else:
                error = model(inputs) - targets.to(device)
                loss = torch.view_as_real(error).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * count
            seen += count
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs} | MSE {total_loss / seen:.6f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.stage}.pt"
    torch.save(model.state_dict(), destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("codec", "importance", "predictor"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--csi", type=Path, default=ROOT / "data" / "csi" / "train.npy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "trained")
    parser.add_argument("--codec", type=Path, help="Teacher weights (default: codec.pt in output directory)")
    parser.add_argument("--epochs", type=int, help="Default: 100 for importance, 200 otherwise")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--snr", type=float, default=13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Use the first N training images or prediction windows")
    args = parser.parse_args()
    if args.codec is None:
        args.codec = args.output_dir / "codec.pt"
    if args.epochs is None:
        args.epochs = 100 if args.stage == "importance" else 200
    train(args)
