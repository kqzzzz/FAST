"""Space-time allocation and flat-fading MIMO transmission."""

import math

import torch


def mmse_filter(channel, noise_ratio):
    eye = torch.eye(channel.shape[-2], dtype=channel.dtype, device=channel.device)
    covariance = channel @ channel.mH + noise_ratio * eye
    return torch.linalg.solve(covariance, channel).mH


def waterfill_amplitudes(singular, noise_ratio):
    """Historical two-stream water filling, with unit power per active stream."""
    floor = noise_ratio / singular.square().clamp_min(1e-20)
    level = 1 + floor.mean(dim=-1, keepdim=True)
    both = (level - floor).clamp_min(0).sqrt()
    strongest = torch.zeros_like(both)
    strongest[..., 0] = 1
    return torch.where(floor[..., -1:] < level, both, strongest)


def resource_quality(predicted, mode, noise_ratio):
    if mode == "svd":
        singular = torch.linalg.svdvals(predicted)
        return singular * waterfill_amplitudes(singular, noise_ratio)
    equalizer = mmse_filter(predicted, noise_ratio)
    gains = (equalizer @ predicted).abs().square()
    signal = gains.diagonal(dim1=-2, dim2=-1)
    # The archived 2 x 2 simulator sums over the first matrix axis.
    interference = (gains.sum(dim=-2) - signal).clamp_min(0)
    noise = noise_ratio * equalizer.abs().square().sum(dim=-2)
    return signal / (interference + noise).clamp_min(1e-12)


def feature_order(importance, quality):
    """Return the feature index assigned to each (time slot, stream) position."""
    ranked_features = importance.argsort(dim=1, descending=True, stable=True)
    ranked_slots = quality.flatten(1).argsort(dim=1, descending=True, stable=True)
    return torch.empty_like(ranked_features).scatter_(1, ranked_slots, ranked_features)


def transmit(features, channel, snr_db, mode, order=None):
    """Channel shape is (images, slots, receive antennas, transmit antennas)."""
    batch, channels, height, width = features.shape
    slots = channel.shape[1]
    streams = channels // slots
    noise_ratio = 10 ** (-snr_db / 10)
    image_ids = torch.arange(batch, device=features.device)[:, None]
    if order is not None:
        features = features[image_ids, order]

    pairs = features.reshape(batch, slots, streams, height * width // 2, 2)
    symbols = torch.view_as_complex(pairs.contiguous())
    if mode != "power_control":
        symbols = symbols / symbols.abs().square().mean().clamp_min(1e-12).sqrt()
    if mode in ("svd", "power_control"):
        left, singular, right_h = torch.linalg.svd(channel, full_matrices=False)
        if mode == "svd":
            amplitudes = waterfill_amplitudes(singular, noise_ratio)
        else:
            inverse = singular.clamp_min(1e-10).reciprocal()
            amplitudes = inverse / inverse.square().mean(dim=-1, keepdim=True).sqrt()
        symbols = right_h.mH @ (amplitudes.unsqueeze(-1) * symbols)
        gain = singular * amplitudes
        inverse_gain = torch.where(gain > 0, gain.clamp_min(1e-10).reciprocal(), 0)
        equalizer = inverse_gain.unsqueeze(-1) * left.mH
    else:
        equalizer = mmse_filter(channel, noise_ratio)

    if mode == "power_control":
        # The codec training channel normalized after precoding and scaled
        # the noise variance by the transmit antenna count.
        symbols = symbols / symbols.abs().square().mean().clamp_min(1e-12).sqrt()
        noise_ratio *= channel.shape[-1]
    received = channel @ symbols
    received = received + math.sqrt(noise_ratio) * torch.randn_like(received)
    decoded = torch.view_as_real((equalizer @ received).resolve_conj()).reshape_as(features)
    if order is not None:
        decoded = decoded[image_ids, order.argsort(dim=1)]
    return decoded
