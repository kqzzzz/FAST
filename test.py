"""Evaluate the pretrained FAST pipeline on CIFAR-10."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from channel import feature_order, resource_quality, transmit
from data import ROOT, csi_windows, load_csi, load_images, load_weights
from models import ChannelPredictor, Codec, WNet, normalize_features


@torch.inference_mode()
def evaluate(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    codec = load_weights(Codec(), args.weights / "codec.pt").to(device).eval()
    evaluator = load_weights(WNet(), args.weights / "importance.pt").to(device).eval()
    predictor = load_weights(ChannelPredictor(), args.weights / "predictor.pt").to(device).eval()

    streams = ChannelPredictor.antennas
    slots = codec.channels // streams
    trace = load_csi(args.csi)
    needed = predictor.history + args.num_images * slots
    if len(trace) < needed:
        raise ValueError(f"This evaluation needs {needed} CSI samples. The trace has {len(trace)}.")
    images = load_images(args.data_dir, train=False, limit=args.num_images)
    loader = DataLoader(images, batch_size=args.batch_size)
    psnr_sum = 0.0
    n_images = 0
    preview = None
    for images, _ in loader:
        history, actual = csi_windows(trace, n_images, len(images), slots)
        images = images.to(device)
        predicted = predictor.forecast(history.to(device), predictor.refresh).reshape_as(actual)
        features = codec.encoder(images)
        importance = evaluator(features, args.evaluator_snr)
        quality = resource_quality(predicted, args.mode, 10 ** (-args.snr / 10))
        order = feature_order(importance, quality)
        received = transmit(normalize_features(features), actual.to(device), args.snr, args.mode, order)
        reconstruction = codec.decoder(received).clamp(0, 1)
        mse = (reconstruction - images).square().mean(dim=(1, 2, 3))
        psnr_sum += (-10 * mse.log10()).sum().item()
        if preview is None:
            preview = torch.stack((images[0], reconstruction[0])).cpu()
        n_images += len(images)

    mean_psnr = psnr_sum / n_images
    if not math.isfinite(mean_psnr):
        raise RuntimeError("Non-finite reconstruction metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(preview, args.output, nrow=2)
    print(f"{args.mode.upper()} | {streams} streams | {args.snr:g} dB | {n_images} images")
    print(f"Mean per-image PSNR: {mean_psnr:.3f} dB")
    print(f"Input and reconstruction: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--csi", type=Path, default=ROOT / "data" / "test_excerpt.npy")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "reconstruction.png")
    parser.add_argument("--mode", choices=("svd", "mmse"), default="svd")
    parser.add_argument("--snr", type=float, default=13, help="Transmission SNR in dB")
    parser.add_argument("--evaluator-snr", type=float, default=13, help="WNet conditioning SNR in dB")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    evaluate(parser.parse_args())
